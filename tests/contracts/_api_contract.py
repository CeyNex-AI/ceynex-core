"""A stable projection of the FastAPI app's OpenAPI spec.

The full `app.openapi()` output is large and churns on prose (summaries,
descriptions, examples), so snapshotting it verbatim would fail on every
docstring edit. This projects it down to the parts a client actually integrates
against: the set of endpoints, and for each request/response body and each
component schema, the property names and which are required. That is the
surface ceynex-web breaks against, and nothing else.
"""

from __future__ import annotations

from typing import Any


def _schema_shape(schema: dict[str, Any]) -> Any:
    """The integration-relevant shape of one JSON schema node: a component
    reference by name, or an object's property names + required set, recursing
    into arrays and compositions but never into prose."""
    if "$ref" in schema:
        return {"$ref": schema["$ref"].split("/")[-1]}
    out: dict[str, Any] = {}
    if "type" in schema:
        out["type"] = schema["type"]
    if "properties" in schema:
        out["properties"] = {k: _schema_shape(v) for k, v in sorted(schema["properties"].items())}
        out["required"] = sorted(schema.get("required", []))
    if "items" in schema:
        out["items"] = _schema_shape(schema["items"])
    for comp in ("anyOf", "oneOf", "allOf"):
        if comp in schema:
            out[comp] = [_schema_shape(s) for s in schema[comp]]
    return out


def project(spec: dict[str, Any]) -> dict[str, Any]:
    """Reduce a full OpenAPI document to its stable contract projection."""
    paths: dict[str, Any] = {}
    for path, ops in spec.get("paths", {}).items():
        for method, op in ops.items():
            if method not in ("get", "post", "put", "patch", "delete"):
                continue
            entry: dict[str, Any] = {}
            params = sorted(
                f"{p['in']}:{p['name']}" + ("*" if p.get("required") else "")
                for p in op.get("parameters", [])
            )
            if params:
                entry["params"] = params
            body = (
                op.get("requestBody", {})
                .get("content", {})
                .get("application/json", {})
                .get("schema")
            )
            if body is not None:
                entry["request"] = _schema_shape(body)
            responses: dict[str, Any] = {}
            for code, resp in sorted(op.get("responses", {}).items()):
                sch = resp.get("content", {}).get("application/json", {}).get("schema")
                responses[code] = _schema_shape(sch) if sch is not None else None
            entry["responses"] = responses
            paths[f"{method.upper()} {path}"] = entry

    schemas = {
        name: _schema_shape(sch)
        for name, sch in sorted(spec.get("components", {}).get("schemas", {}).items())
    }
    return {"paths": paths, "schemas": schemas}


def current_projection() -> dict[str, Any]:
    from ceynex.api.main import app

    return project(app.openapi())
