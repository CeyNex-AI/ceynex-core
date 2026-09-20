"""A runtime check that a value satisfies a frozen `ceynex.contracts` TypedDict.

TypedDicts are erased at runtime, so nothing normally stops an agent from
drifting away from the shape the merger, the API and ceynex-web all read. These
helpers close that gap for the contract tests: they read the *current* contract's
`__required_keys__` and `__annotations__`, so if someone edits `ceynex/contracts`
the producers here are re-checked against the new shape and core CI fails, rather
than the drift surfacing as a KeyError in ceynex-web at runtime.

Deliberately small: it covers the shapes CeyNex actually passes across the
member boundary (str, float, bool, list[...], dict[str, float], and nested
TypedDicts), not the whole typing module.
"""

from __future__ import annotations

import typing
from typing import Any, get_args, get_origin


def _check(value: Any, hint: Any, path: str, errors: list[str]) -> None:
    # NotRequired[...] / Required[...] unwrap to their argument.
    origin = get_origin(hint)
    if origin is typing.Required or origin is typing.NotRequired:
        _check(value, get_args(hint)[0], path, errors)
        return

    if hint is float:
        # bool is an int, int is an acceptable float; reject nothing numeric.
        if isinstance(value, bool) or not isinstance(value, int | float):
            errors.append(f"{path}: expected float, got {type(value).__name__}")
        return
    if hint in (str, bool, int):
        if not isinstance(value, hint) or (hint is int and isinstance(value, bool)):
            errors.append(f"{path}: expected {hint.__name__}, got {type(value).__name__}")
        return

    if origin is list:
        if not isinstance(value, list):
            errors.append(f"{path}: expected list, got {type(value).__name__}")
            return
        (item_hint,) = get_args(hint) or (Any,)
        for i, item in enumerate(value):
            _check(item, item_hint, f"{path}[{i}]", errors)
        return

    if origin is dict:
        if not isinstance(value, dict):
            errors.append(f"{path}: expected dict, got {type(value).__name__}")
            return
        key_hint, val_hint = get_args(hint) or (Any, Any)
        for k, v in value.items():
            _check(k, key_hint, f"{path}.<key>", errors)
            _check(v, val_hint, f"{path}[{k!r}]", errors)
        return

    # A nested TypedDict (Evidence, ForecastPoint) — recurse.
    if is_typeddict(hint):
        assert_matches(value, hint, path, errors)
        return

    # Anything else (Any, Literal, unions we do not model): accept.


def is_typeddict(hint: Any) -> bool:
    return hasattr(hint, "__required_keys__") and hasattr(hint, "__annotations__")


def assert_matches(
    value: Any, td: Any, path: str = "", errors: list[str] | None = None
) -> list[str]:
    """Collect every way `value` fails to satisfy the TypedDict `td`.

    Returns the error list (empty when it conforms) so a test can assert on it
    with a readable message rather than a bare AssertionError.
    """
    top = errors is None
    if errors is None:
        errors = []
    here = path or td.__name__

    if not isinstance(value, dict):
        errors.append(f"{here}: expected a dict for {td.__name__}, got {type(value).__name__}")
        return errors

    hints = typing.get_type_hints(td, include_extras=True)
    for key in td.__required_keys__:
        if key not in value:
            errors.append(f"{here}: missing required key {key!r}")
        else:
            _check(value[key], hints[key], f"{here}.{key}", errors)
    for key in getattr(td, "__optional_keys__", frozenset()):
        if key in value:
            _check(value[key], hints[key], f"{here}.{key}", errors)

    return errors if top else errors
