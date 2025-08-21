# askmyapi

[![builds.sr.ht status](https://builds.sr.ht/~jcdenton/askmyapi.svg)](https://builds.sr.ht/~jcdenton/askmyapi?)
[![License: CC0-1.0](https://img.shields.io/badge/license-CC0%201.0-blue.svg)](https://creativecommons.org/publicdomain/zero/1.0/)

Query (almost) any OpenAPI/Swagger specification through a RAG-powered assistant.

## Why
API docs are fragmented across operations, schemas, and examples. AskMyAPI ingests an OpenAPI spec (JSON or YAML), resolves all `$ref`s, and builds a multi-vector index so you can ask natural questions and get precise, runnable answers (including `curl`).

## Features
- OpenAPI JSON/YAML loading
- $ref resolution (internal & external)
- Documents per operation, parameters, request bodies, responses, and schemas
- Rich metadata (method, path, operationId, tag, status_code…)
- Multi-vector RAG: summaries, HyDE-style questions, and example requests
- Multilingual embeddings (default: `intfloat/multilingual-e5-base`)
- Gradio chat UI with live indexing and memory
- Storage with Chroma, restart-safe and cached by spec hash

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Environment
Create a `.env` (optional):
```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
OLLAMA_MODEL=llama3
GRADIO_SERVER_NAME=127.0.0.1
GRADIO_SERVER_PORT=7860
GRADIO_SHARE=0
```
If `OPENAI_API_KEY` is not set, AskMyAPI falls back to Ollama.

## Usage
```bash
askmyapi /path/to/openapi.{json|yaml|yml}
```
This will:
1. Load and dereference the spec.
2. Build or resume the Chroma index for this spec.
3. Launch a Gradio chat at the given port.

### Chat tips
- Ask: “How do I create a pet?” – The answer includes method/path, required params, and a ready-to-run `curl`.
- Upload additional plain text / Markdown from the UI to extend the index on the fly.

## Project structure
```
askmyapi/
  src/askmyapi/
    config.py
    spec_loader.py
    ingestion.py
    vectorstore.py
    rag.py
    interface.py
    __main__.py
```

## Design notes
- Deterministic IDs using `operationId` when available (falling back to content hash).
- Spec-scoped caches (`*_summaries.json`) keyed by a spec hash so multiple APIs never collide.
- Metadata-first retrieval enables filtering or advanced routing in future versions.

## Limitations
- Large specs can take time to index on first run (LLM-generated summaries/examples).
- Example generation is heuristic; always validate against the real API.