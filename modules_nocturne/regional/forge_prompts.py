"""Lazy bridges to Forge's supported prompt parsing APIs."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from modules_nocturne.regional.prompts import CompiledPromptPlan, CompiledPromptText, PromptOwner


@dataclass(frozen=True, slots=True)
class ForgeScheduleEntry:
    end_at_step: int
    text: str


@dataclass(frozen=True, slots=True)
class ForgePromptSchedule:
    owner: PromptOwner
    polarity: str
    entries: tuple[ForgeScheduleEntry, ...]


@dataclass(frozen=True, slots=True)
class ForgeExtraNetworkParse:
    cleaned_positive_prompts: Mapping[PromptOwner, str]
    extra_network_data: Mapping[str, tuple[Any, ...]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "cleaned_positive_prompts", MappingProxyType(dict(self.cleaned_positive_prompts)))
        object.__setattr__(self, "extra_network_data", MappingProxyType(dict(self.extra_network_data)))


def _all_prompts(compiled: CompiledPromptPlan) -> tuple[CompiledPromptText, ...]:
    prompts = [compiled.global_positive, compiled.global_negative]
    for region in compiled.regions:
        prompts.extend((region.positive, region.negative))
    return tuple(prompts)


def resolve_forge_schedules(
    compiled: CompiledPromptPlan,
    *,
    steps: int,
    hires_steps: int | None = None,
) -> tuple[ForgePromptSchedule, ...]:
    """Resolve scheduling and alternation through Forge's parser."""

    if steps < 1 or hires_steps is not None and hires_steps < 1:
        raise ValueError("Prompt schedule step counts must be positive")
    from modules import prompt_parser

    prompts = _all_prompts(compiled)
    schedules = prompt_parser.get_learned_conditioning_prompt_schedules(
        [prompt.text for prompt in prompts],
        steps,
        hires_steps,
    )
    return tuple(
        ForgePromptSchedule(
            owner=prompt.owner,
            polarity=prompt.polarity,
            entries=tuple(ForgeScheduleEntry(end_at_step=int(end), text=text) for end, text in schedule),
        )
        for prompt, schedule in zip(prompts, schedules)
    )


def parse_forge_extra_networks(compiled: CompiledPromptPlan) -> ForgeExtraNetworkParse:
    """Parse and deduplicate generation-global tags through Forge's parser."""

    from modules import extra_networks

    positive = [compiled.global_positive, *(region.positive for region in compiled.regions)]
    cleaned: dict[PromptOwner, str] = {}
    deduplicated: dict[str, list[Any]] = {}
    seen: set[tuple[str, tuple[Any, ...]]] = set()
    for prompt in positive:
        cleaned_text, parsed = extra_networks.parse_prompt(prompt.text)
        cleaned[prompt.owner] = cleaned_text
        for network_name, parameters in parsed.items():
            for parameter in parameters:
                key = (network_name, tuple(parameter.items))
                if key in seen:
                    continue
                seen.add(key)
                deduplicated.setdefault(network_name, []).append(parameter)
    return ForgeExtraNetworkParse(
        cleaned_positive_prompts=cleaned,
        extra_network_data={name: tuple(parameters) for name, parameters in deduplicated.items()},
    )
