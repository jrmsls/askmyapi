import json
from typing import List, Dict, Any
from langchain.schema import Document

_HTTP_METHODS = {
    "get",
    "put",
    "post",
    "delete",
    "options",
    "head",
    "patch",
    "trace",
}


def _doc(content: str, **meta) -> Document:
    """Helper to build a Document with metadata stripped of empty values."""
    metadata = {k: v for k, v in meta.items() if v not in (None, "", [], {})}
    return Document(page_content=content, metadata=metadata)


def openapi_to_documents(
    spec: Dict[str, Any], api_name: str, spec_hash: str
) -> List[Document]:
    """
    Transform a dereferenced OpenAPI specification into a list of Documents.
    We create:
      - One parent "operation" document per (method, path)
      - Child documents for parameters, request bodies, responses
      - Schema documents for top-level components.schemas
    """
    docs: List[Document] = []

    servers = spec.get("servers", [])
    base_urls = [
        s.get("url") for s in servers if isinstance(s, dict) and s.get("url")
    ]

    paths = spec.get("paths", {}) or {}
    for path, item in paths.items():
        if not isinstance(item, dict):
            continue
        for method, op in item.items():
            if method.lower() not in _HTTP_METHODS or not isinstance(op, dict):
                continue

            op_id = op.get("operationId") or f"{method}_{path}".replace(
                "/", "_"
            )
            tags = op.get("tags", [])
            summary = op.get("summary", "")
            description = op.get("description", "")

            # ---- Parent: operation
            parent_txt = [
                f"OPERATION: {method.upper()} {path}",
                f"OPERATION_ID: {op_id}",
                f"TAGS: {', '.join(tags) if tags else 'N/A'}",
                f"SUMMARY: {summary or 'N/A'}",
                "DESCRIPTION:",
                description or "N/A",
                f"BASE_URLS: {', '.join(base_urls) or 'N/A'}",
            ]
            docs.append(
                _doc(
                    "\n".join(parent_txt),
                    kind="operation",
                    api_name=api_name,
                    spec_hash=spec_hash,
                    method=method.upper(),
                    path=path,
                    operationId=op_id,
                    tags=tags,
                )
            )

            # ---- Children: parameters
            for p in op.get("parameters", []) or []:
                name = p.get("name")
                loc = p.get("in")
                required = p.get("required", False)
                desc = p.get("description", "")
                schema = p.get("schema", {})
                docs.append(
                    _doc(
                        "PARAMETER\n"
                        f"for: {method.upper()} {path}\n"
                        f"name: {name}\n"
                        f"in: {loc}\n"
                        f"required: {required}\n"
                        f"description: {desc}\n"
                        f"schema: {json.dumps(schema, ensure_ascii=False, indent=2)}",
                        kind="parameter",
                        api_name=api_name,
                        spec_hash=spec_hash,
                        method=method.upper(),
                        path=path,
                        operationId=op_id,
                        param_in=loc,
                        param_name=name,
                        required=required,
                    )
                )

            # ---- Child: requestBody
            if "requestBody" in op:
                rb = op["requestBody"] or {}
                content = rb.get("content", {})
                docs.append(
                    _doc(
                        "REQUEST BODY\n"
                        f"for: {method.upper()} {path}\n"
                        f"required: {rb.get('required', False)}\n"
                        f"content: {json.dumps(content, ensure_ascii=False, indent=2)}",
                        kind="requestBody",
                        api_name=api_name,
                        spec_hash=spec_hash,
                        method=method.upper(),
                        path=path,
                        operationId=op_id,
                    )
                )

            # ---- Children: responses per status
            for status, resp in (op.get("responses") or {}).items():
                if not isinstance(resp, dict):
                    continue
                desc = resp.get("description", "")
                content = resp.get("content", {})
                docs.append(
                    _doc(
                        "RESPONSE\n"
                        f"for: {method.upper()} {path}\n"
                        f"status: {status}\n"
                        f"description: {desc}\n"
                        f"content: {json.dumps(content, ensure_ascii=False, indent=2)}",
                        kind="response",
                        api_name=api_name,
                        spec_hash=spec_hash,
                        method=method.upper(),
                        path=path,
                        operationId=op_id,
                        status_code=status,
                    )
                )

    # ---- Components: schemas
    schemas = (spec.get("components", {}) or {}).get("schemas", {}) or {}
    for name, schema in schemas.items():
        title = schema.get("title", name)
        desc = schema.get("description", "")
        docs.append(
            _doc(
                "SCHEMA\n"
                f"name: {name}\n"
                f"title: {title}\n"
                f"description:\n{desc}\n"
                f"schema_json:\n{json.dumps(schema, ensure_ascii=False, indent=2)}",
                kind="schema",
                api_name=api_name,
                spec_hash=spec_hash,
                schema_name=name,
            )
        )

    return docs
