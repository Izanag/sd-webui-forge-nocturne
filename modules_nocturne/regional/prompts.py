"""Immutable prompt compilation records before model-specific encoding."""

from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from modules_nocturne.regional.model import RegionalGenerationPlan


@dataclass(frozen=True, slots=True)
class PromptExpansionContext:
    base_seed: int
    batch_index: int
    region_id: UUID | None
    polarity: str


class PromptExpansionService(Protocol):
    def expand(self, text: str, context: PromptExpansionContext) -> str: ...


@dataclass(frozen=True, slots=True)
class PromptTokenUsage:
    token_count: int
    token_limit: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.token_count, bool)
            or isinstance(self.token_limit, bool)
            or not isinstance(self.token_count, int)
            or not isinstance(self.token_limit, int)
            or self.token_count < 0
            or self.token_limit < 0
        ):
            raise ValueError("Prompt token counts must be non-negative")

    @property
    def truncated_tokens(self) -> int:
        return max(self.token_count - self.token_limit, 0)


class PromptTokenCounter(Protocol):
    def measure(self, text: str, *, polarity: str) -> PromptTokenUsage: ...


class IdentityPromptExpansion:
    def expand(self, text: str, context: PromptExpansionContext) -> str:
        return text


@dataclass(frozen=True, slots=True)
class PromptOwner:
    kind: str
    region_id: UUID | None = None


@dataclass(frozen=True, slots=True)
class CompiledPromptText:
    owner: PromptOwner
    polarity: str
    source_global: str
    source_local: str
    inherits_global: bool
    text: str
    extra_network_effect: str = "global"
    truncation_tokens: int | None = None
    token_count: int | None = None
    token_limit: int | None = None


@dataclass(frozen=True, slots=True)
class CompiledRegionPrompts:
    region_id: UUID
    positive: CompiledPromptText
    negative: CompiledPromptText


@dataclass(frozen=True, slots=True)
class EncodingRequest:
    polarity: str
    text: str
    owners: tuple[PromptOwner, ...]


@dataclass(frozen=True, slots=True)
class CompiledPromptPlan:
    global_positive: CompiledPromptText
    global_negative: CompiledPromptText
    regions: tuple[CompiledRegionPrompts, ...]
    encoding_requests: tuple[EncodingRequest, ...]


def _compose(global_text: str, local_text: str, inherits_global: bool) -> str:
    parts = []
    if inherits_global and global_text.strip():
        parts.append(global_text)
    if local_text.strip():
        parts.append(local_text)
    return ", ".join(parts)


def compile_prompt_plan(
    plan: RegionalGenerationPlan,
    *,
    base_seed: int,
    batch_index: int = 0,
    expansion_service: PromptExpansionService | None = None,
    token_counter: PromptTokenCounter | None = None,
) -> CompiledPromptPlan:
    expansion = expansion_service or IdentityPromptExpansion()

    def compile_text(
        *,
        owner: PromptOwner,
        polarity: str,
        global_text: str,
        local_text: str,
        inherits_global: bool,
    ) -> CompiledPromptText:
        composed = _compose(global_text, local_text, inherits_global)
        expanded = expansion.expand(
            composed,
            PromptExpansionContext(
                base_seed=base_seed,
                batch_index=batch_index,
                region_id=owner.region_id,
                polarity=polarity,
            ),
        )
        usage = token_counter.measure(expanded, polarity=polarity) if token_counter is not None else None
        return CompiledPromptText(
            owner=owner,
            polarity=polarity,
            source_global=global_text,
            source_local=local_text,
            inherits_global=inherits_global,
            text=expanded,
            truncation_tokens=usage.truncated_tokens if usage is not None else None,
            token_count=usage.token_count if usage is not None else None,
            token_limit=usage.token_limit if usage is not None else None,
        )

    global_owner = PromptOwner(kind="global")
    global_positive = compile_text(
        owner=global_owner,
        polarity="positive",
        global_text=plan.global_prompt.positive,
        local_text="",
        inherits_global=True,
    )
    global_negative = compile_text(
        owner=global_owner,
        polarity="negative",
        global_text=plan.global_prompt.negative,
        local_text="",
        inherits_global=True,
    )

    compiled_regions = []
    for region in plan.regions:
        owner = PromptOwner(kind="region", region_id=region.id)
        compiled_regions.append(
            CompiledRegionPrompts(
                region_id=region.id,
                positive=compile_text(
                    owner=owner,
                    polarity="positive",
                    global_text=plan.global_prompt.positive,
                    local_text=region.positive,
                    inherits_global=region.inherit_global_positive,
                ),
                negative=compile_text(
                    owner=owner,
                    polarity="negative",
                    global_text=plan.global_prompt.negative,
                    local_text=region.negative,
                    inherits_global=region.inherit_global_negative,
                ),
            )
        )

    grouped: dict[tuple[str, str], list[PromptOwner]] = {}
    all_prompts = [global_positive, global_negative]
    for compiled_region in compiled_regions:
        all_prompts.extend((compiled_region.positive, compiled_region.negative))
    for prompt in all_prompts:
        grouped.setdefault((prompt.polarity, prompt.text), []).append(prompt.owner)

    encoding_requests = tuple(
        EncodingRequest(polarity=polarity, text=text, owners=tuple(owners))
        for (polarity, text), owners in sorted(grouped.items(), key=lambda item: item[0])
    )

    return CompiledPromptPlan(
        global_positive=global_positive,
        global_negative=global_negative,
        regions=tuple(compiled_regions),
        encoding_requests=encoding_requests,
    )
