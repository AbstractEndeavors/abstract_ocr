"""
Step registry for the OCR pipeline.

Each step is a callable with a declared (input_type, output_type).
The orchestrator chains steps by matching types — no implicit coupling.
"""
from .imports import (
    annotations,
    dataclass,
    field,
    Any,
    Callable,
    Protocol,
    runtime_checkable
    )


@runtime_checkable
class PipelineStep(Protocol):
    """What every step must look like."""
    name: str
    def __call__(self, data: Any, config: Any) -> Any: ...


@dataclass
class StepEntry:
    name: str
    callable: Callable
    input_type: type
    output_type: type
    description: str = ""


class StepRegistry:
    """
    Explicit registry of pipeline steps.

    Steps are registered with their type signature so the orchestrator
    can validate the chain at build time, not at blow-up time.
    """

    def __init__(self) -> None:
        self._steps: dict[str, StepEntry] = {}

    def register(
        self,
        name: str,
        *,
        input_type: type,
        output_type: type,
        description: str = "",
    ) -> Callable:
        """Decorator that registers a step."""
        def decorator(fn: Callable) -> Callable:
            fn.name = name  # type: ignore[attr-defined]
            self._steps[name] = StepEntry(
                name=name,
                callable=fn,
                input_type=input_type,
                output_type=output_type,
                description=description,
            )
            return fn
        return decorator

    def get(self, name: str) -> StepEntry:
        if name not in self._steps:
            registered = ", ".join(self._steps) or "(none)"
            raise KeyError(
                f"Step {name!r} not in registry. Registered: {registered}"
            )
        return self._steps[name]

    def list_steps(self) -> list[str]:
        return list(self._steps)

    def validate_chain(self, step_names: list[str]) -> list[str]:
        """
        Check that output_type[i] matches input_type[i+1] for the chain.
        Returns list of problems (empty = valid).
        """
        problems: list[str] = []
        for i in range(len(step_names) - 1):
            curr = self.get(step_names[i])
            nxt = self.get(step_names[i + 1])
            if not issubclass(curr.output_type, nxt.input_type):
                problems.append(
                    f"{curr.name} outputs {curr.output_type.__name__} "
                    f"but {nxt.name} expects {nxt.input_type.__name__}"
                )
        return problems


# Module-level registry — steps import this and register themselves.
registry = StepRegistry()
