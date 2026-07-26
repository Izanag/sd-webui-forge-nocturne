"""Adapter and engine capability contracts with an honest unsupported default."""

from dataclasses import dataclass
from threading import RLock
from typing import Any, Protocol, runtime_checkable

from modules_nocturne.regional.model import OverlapPolicy, UncoveredPolicy


@dataclass(frozen=True, slots=True)
class EngineCapabilities:
    engine_id: str
    engine_version: str
    overlap_policies: frozenset[OverlapPolicy]
    uncovered_policies: frozenset[UncoveredPolicy]
    supported_fields: frozenset[str] = frozenset()
    expected_fallbacks: tuple[str, ...] = ()
    cost_warning: str | None = None


@runtime_checkable
class RegionalEngine(Protocol):
    engine_id: str

    def capabilities(self) -> EngineCapabilities: ...


@runtime_checkable
class InstalledRegionalEngine(Protocol):
    def bind_conditioning(self, conditioning: Any) -> None: ...

    def close(self) -> None: ...


@runtime_checkable
class RegionalModelAdapter(Protocol):
    adapter_id: str

    def matches(self, model_context: Any) -> bool: ...

    def supported_engine_ids(self) -> frozenset[str]: ...

    def unsupported_fields(self) -> frozenset[str]: ...

    def expected_fallbacks(self) -> tuple[str, ...]: ...


@dataclass(frozen=True, slots=True)
class CapabilityReport:
    status: str
    adapter_id: str | None = None
    eligible_engines: tuple[EngineCapabilities, ...] = ()
    unsupported_fields: tuple[str, ...] = ()
    expected_fallbacks: tuple[str, ...] = ()
    reason_code: str | None = None
    reason: str | None = None


class EngineRegistry:
    def __init__(self) -> None:
        self._engines: dict[str, RegionalEngine] = {}
        self._lock = RLock()

    def register(self, engine: RegionalEngine) -> None:
        with self._lock:
            if engine.engine_id in self._engines:
                raise ValueError(f"Regional engine {engine.engine_id!r} is already registered")
            self._engines[engine.engine_id] = engine

    def get(self, engine_id: str) -> RegionalEngine | None:
        with self._lock:
            return self._engines.get(engine_id)

    def clear(self) -> None:
        with self._lock:
            self._engines.clear()


class AdapterRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, RegionalModelAdapter] = {}
        self._lock = RLock()

    def register(self, adapter: RegionalModelAdapter) -> None:
        with self._lock:
            if adapter.adapter_id in self._adapters:
                raise ValueError(f"Regional adapter {adapter.adapter_id!r} is already registered")
            self._adapters[adapter.adapter_id] = adapter

    def matching(self, model_context: Any) -> RegionalModelAdapter | None:
        with self._lock:
            adapters = tuple(self._adapters.values())
        for adapter in adapters:
            if adapter.matches(model_context):
                return adapter
        return None

    def get(self, adapter_id: str) -> RegionalModelAdapter | None:
        with self._lock:
            return self._adapters.get(adapter_id)

    def clear(self) -> None:
        with self._lock:
            self._adapters.clear()


class CapabilityService:
    def __init__(
        self,
        *,
        adapters: AdapterRegistry | None = None,
        engines: EngineRegistry | None = None,
    ) -> None:
        self.adapters = adapters or AdapterRegistry()
        self.engines = engines or EngineRegistry()

    def report(self, model_context: Any) -> CapabilityReport:
        if model_context is None:
            return CapabilityReport(
                status="unsupported",
                reason_code="model.not_loaded",
                reason="No model is loaded",
            )

        adapter = self.adapters.matching(model_context)
        if adapter is None:
            return CapabilityReport(
                status="unsupported",
                reason_code="model.adapter_not_found",
                reason="No registered Regional adapter supports the current model",
            )

        eligible = []
        missing_engines = []
        for engine_id in sorted(adapter.supported_engine_ids()):
            engine = self.engines.get(engine_id)
            if engine is None:
                missing_engines.append(engine_id)
                continue
            eligible.append(engine.capabilities())

        fallbacks = list(adapter.expected_fallbacks())
        if missing_engines:
            fallbacks.append(f"Unavailable registered engines: {', '.join(missing_engines)}")

        if not eligible:
            return CapabilityReport(
                status="unsupported",
                adapter_id=adapter.adapter_id,
                unsupported_fields=tuple(sorted(adapter.unsupported_fields())),
                expected_fallbacks=tuple(fallbacks),
                reason_code="engine.none_eligible",
                reason="The matched adapter has no available Regional engine",
            )

        return CapabilityReport(
            status="supported",
            adapter_id=adapter.adapter_id,
            eligible_engines=tuple(eligible),
            unsupported_fields=tuple(sorted(adapter.unsupported_fields())),
            expected_fallbacks=tuple(fallbacks),
        )


capability_service = CapabilityService()
