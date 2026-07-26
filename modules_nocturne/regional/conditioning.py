"""Forge-backed Regional conditioning with immutable owner mappings."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from modules_nocturne.regional.forge_prompts import (
    ForgePromptSchedule,
    ForgeScheduleEntry,
    parse_forge_extra_networks,
    resolve_forge_schedules,
)
from modules_nocturne.regional.prompts import CompiledPromptPlan, PromptOwner


@dataclass(frozen=True, slots=True)
class WeightedConditioning:
    text: str
    weight: float
    value: Any = None


@dataclass(frozen=True, slots=True)
class RegionalConditioningScheduleEntry:
    end_at_step: int
    resolved_text: str
    encoded: tuple[WeightedConditioning, ...]


@dataclass(frozen=True, slots=True)
class RegionalPromptConditioning:
    owner: PromptOwner
    polarity: str
    entries: tuple[RegionalConditioningScheduleEntry, ...]


@dataclass(frozen=True, slots=True)
class RegionalImageConditioning:
    image_index: int
    prompts: Mapping[tuple[PromptOwner, str], RegionalPromptConditioning]

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompts", MappingProxyType(dict(self.prompts)))

    def get(self, owner: PromptOwner, polarity: str) -> RegionalPromptConditioning:
        return self.prompts[(owner, polarity)]


@dataclass(frozen=True, slots=True)
class FinalPromptRecord:
    image_index: int
    owner: PromptOwner
    polarity: str
    entries: tuple[ForgeScheduleEntry, ...]


@dataclass(frozen=True, slots=True)
class RegionalConditioningBatch:
    images: tuple[RegionalImageConditioning, ...]
    encoded_texts: tuple[tuple[str, str], ...]
    final_prompts: tuple[FinalPromptRecord, ...]
    extra_network_data: Mapping[str, tuple[Any, ...]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "images", tuple(self.images))
        object.__setattr__(self, "encoded_texts", tuple(self.encoded_texts))
        object.__setattr__(self, "final_prompts", tuple(self.final_prompts))
        object.__setattr__(self, "extra_network_data", MappingProxyType(dict(self.extra_network_data)))

    @property
    def encoding_count(self) -> int:
        return len(self.encoded_texts)


def _merge_extra_network_data(plans: tuple[CompiledPromptPlan, ...]) -> Mapping[str, tuple[Any, ...]]:
    merged: dict[str, list[Any]] = {}
    seen: set[tuple[str, tuple[Any, ...]]] = set()
    for plan in plans:
        parsed = parse_forge_extra_networks(plan)
        for network_name, parameters in parsed.extra_network_data.items():
            for parameter in parameters:
                key = (network_name, tuple(parameter.items))
                if key in seen:
                    continue
                seen.add(key)
                merged.setdefault(network_name, []).append(parameter)
    return MappingProxyType({name: tuple(parameters) for name, parameters in merged.items()})


def _positive_parts(text: str) -> tuple[tuple[str, float], ...]:
    from modules import extra_networks, prompt_parser

    cleaned, _ = extra_networks.parse_prompt(text)
    indexes, flat, _ = prompt_parser.get_multicond_prompt_list([cleaned])
    return tuple((str(flat[index]), float(weight)) for index, weight in indexes[0])


def _schedule_parts(schedule: ForgePromptSchedule) -> tuple[tuple[tuple[str, float], ...], ...]:
    if schedule.polarity == "positive":
        return tuple(_positive_parts(entry.text) for entry in schedule.entries)
    return tuple(((entry.text, 1.0),) for entry in schedule.entries)


def _split_encoded_batch(encoded: Any, count: int) -> tuple[Any, ...]:
    if isinstance(encoded, dict):
        lengths = {len(value) for value in encoded.values()}
        if lengths != {count}:
            raise RuntimeError("Forge returned an unexpected dictionary conditioning batch size")
        return tuple({key: value[index] for key, value in encoded.items()} for index in range(count))
    if len(encoded) != count:
        raise RuntimeError("Forge returned an unexpected conditioning batch size")
    return tuple(encoded[index] for index in range(count))


def build_regional_conditioning(
    compiled_plans: tuple[CompiledPromptPlan, ...],
    *,
    image_indices: tuple[int, ...],
    model_context: Any,
    steps: int,
    width: int,
    height: int,
    distilled_cfg_scale: float,
    hires_steps: int | None = None,
) -> RegionalConditioningBatch:
    """Encode unique scheduled strings through the active Forge text encoder."""

    if not compiled_plans or len(compiled_plans) != len(image_indices):
        raise ValueError("Compiled Regional prompts must align one-to-one with image indices")
    if len(set(image_indices)) != len(image_indices) or any(
        isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in image_indices
    ):
        raise ValueError("Regional conditioning image indices must be unique non-negative integers")

    from modules import prompt_parser

    schedules_by_image = tuple(
        resolve_forge_schedules(plan, steps=steps, hires_steps=hires_steps)
        for plan in compiled_plans
    )
    parts_by_image = tuple(
        tuple(_schedule_parts(schedule) for schedule in schedules)
        for schedules in schedules_by_image
    )

    unique_texts: dict[str, list[str]] = {"positive": [], "negative": []}
    seen_texts: set[tuple[str, str]] = set()
    for schedules, schedule_parts in zip(schedules_by_image, parts_by_image):
        for schedule, entries in zip(schedules, schedule_parts):
            for parts in entries:
                for text, _weight in parts:
                    key = schedule.polarity, text
                    if key in seen_texts:
                        continue
                    seen_texts.add(key)
                    unique_texts[schedule.polarity].append(text)

    encoded_by_key: dict[tuple[str, str], Any] = {}
    for polarity in ("positive", "negative"):
        texts = unique_texts[polarity]
        if not texts:
            continue
        forge_prompts = prompt_parser.SdConditioning(
            texts,
            is_negative_prompt=polarity == "negative",
            width=width,
            height=height,
            distilled_cfg_scale=distilled_cfg_scale,
        )
        encoded = model_context.get_learned_conditioning(forge_prompts)
        for text, value in zip(texts, _split_encoded_batch(encoded, len(texts))):
            encoded_by_key[(polarity, text)] = value

    images = []
    final_prompts = []
    for image_index, schedules, schedule_parts in zip(image_indices, schedules_by_image, parts_by_image):
        prompt_mapping = {}
        for schedule, entries in zip(schedules, schedule_parts):
            compiled_entries = tuple(
                RegionalConditioningScheduleEntry(
                    end_at_step=schedule_entry.end_at_step,
                    resolved_text=schedule_entry.text,
                    encoded=tuple(
                        WeightedConditioning(
                            text=text,
                            weight=weight,
                            value=encoded_by_key[(schedule.polarity, text)],
                        )
                        for text, weight in parts
                    ),
                )
                for schedule_entry, parts in zip(schedule.entries, entries)
            )
            prompt_mapping[(schedule.owner, schedule.polarity)] = RegionalPromptConditioning(
                owner=schedule.owner,
                polarity=schedule.polarity,
                entries=compiled_entries,
            )
            final_prompts.append(
                FinalPromptRecord(
                    image_index=image_index,
                    owner=schedule.owner,
                    polarity=schedule.polarity,
                    entries=schedule.entries,
                )
            )
        images.append(RegionalImageConditioning(image_index=image_index, prompts=prompt_mapping))

    return RegionalConditioningBatch(
        images=tuple(images),
        encoded_texts=tuple((polarity, text) for polarity in ("positive", "negative") for text in unique_texts[polarity]),
        final_prompts=tuple(final_prompts),
        extra_network_data=_merge_extra_network_data(compiled_plans),
    )
