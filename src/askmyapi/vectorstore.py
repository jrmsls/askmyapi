import os
import hashlib
import json
import logging
from typing import Dict, Set, Tuple, List

from langchain_chroma import Chroma
from langchain.storage import InMemoryByteStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from tenacity import retry, wait_exponential, stop_after_attempt
from tqdm import tqdm

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "intfloat/multilingual-e5-base"  # Fixed choice: robust multilingual retrieval

# Directories for persistence
CHROMA_DIR = "chroma_store"
CACHE_DIR = "cache"
os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


def _safe_content(msg_or_str) -> str:
    """Return a string content from a ChatMessage or raw string."""
    return getattr(msg_or_str, "content", msg_or_str)


def setup_vectorstore(
    documents: List,
    llm,
    *,
    api_name: str,
    spec_hash: str,
) -> Tuple[MultiVectorRetriever, InMemoryByteStore]:
    """
    Build or resume the Chroma multivector index from a list of parent/child Documents.
    - Deterministic IDs per document to keep re-indexing stable.
    - Spec-scoped caches (summaries, questions, examples) to avoid mixing APIs.
    """
    logger.info(
        "Setting up vectorstore for API=%s, hash=%s", api_name, spec_hash
    )

    docstore = InMemoryByteStore()
    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    COLLECTION = f"openapi_vectors_{api_name}_{spec_hash}"

    SUMMARY_CACHE_PATH = os.path.join(
        CACHE_DIR, f"{COLLECTION}_summaries.json"
    )
    QUESTIONS_CACHE_PATH = os.path.join(
        CACHE_DIR, f"{COLLECTION}_questions.json"
    )
    EXAMPLES_CACHE_PATH = os.path.join(
        CACHE_DIR, f"{COLLECTION}_examples.json"
    )

    summaries: Dict[str, str] = _load_json(SUMMARY_CACHE_PATH)
    questions: Dict[str, str] = _load_json(QUESTIONS_CACHE_PATH)
    examples: Dict[str, str] = _load_json(EXAMPLES_CACHE_PATH)

    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION,
        embedding_function=embedding,
    )

    existing_ids: Set[str] = _get_all_ids(vectorstore)
    logger.debug("Vectorstore already contains %d IDs", len(existing_ids))

    @retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3))
    def _llm(txt: str) -> str:
        """Small retry wrapper to improve robustness on LLM calls."""
        msg = llm.invoke(txt)
        return _safe_content(msg)

    def generate_summary(text: str) -> str:
        return _llm(
            "Summarize this API doc chunk in <=6 lines. "
            "Focus on method/path, required parameters, and purpose:\n" + text
        )

    def generate_questions(text: str) -> str:
        return _llm(
            "Generate 4 likely user questions about this API chunk (bulleted, concise):\n"
            + text
        )

    def generate_examples(text: str) -> str:
        return _llm(
            "From this API description, produce a minimal curl example if applicable. "
            "Include method, path, required params, and a placeholder base URL if absent. "
            "If no HTTP call is relevant, return 'N/A'.\n" + text
        )

    for i, doc in enumerate(tqdm(documents, desc="Indexing")):
        op_id = doc.metadata.get("operationId")
        stable = (
            op_id
            or hashlib.sha1(doc.page_content.encode("utf-8")).hexdigest()[:12]
        )
        base_id = f"{doc.metadata.get('kind','doc')}::{stable}::{i}"

        logger.debug(
            "Indexing doc #%d with base_id=%s, kind=%s",
            i,
            base_id,
            doc.metadata.get("kind"),
        )

        # Store the raw parent/child doc in the byte store.
        docstore.mset([(base_id, doc)])

        # Derived texts
        for kind, cache, gen in [
            ("summary", summaries, generate_summary),
            ("hyde", questions, generate_questions),
            ("example", examples, generate_examples),
        ]:
            child_id = f"{base_id}:{kind}"
            if child_id in existing_ids:
                logger.debug("Skipping existing child_id=%s", child_id)
                continue
            logger.debug("Generating %s for %s", kind, base_id)
            text = cache.get(base_id) or gen(doc.page_content)
            cache[base_id] = text
            vectorstore.add_texts(
                [text],
                ids=[child_id],
                metadatas=[{"doc_id": base_id, "kind": kind, **doc.metadata}],
            )

    _dump_json(SUMMARY_CACHE_PATH, summaries)
    _dump_json(QUESTIONS_CACHE_PATH, questions)
    _dump_json(EXAMPLES_CACHE_PATH, examples)

    logger.info(
        "Finished indexing. Collection=%s, total_docs=%d",
        COLLECTION,
        len(documents),
    )

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key="doc_id",
    )
    return retriever, docstore


def _get_all_ids(vectorstore: Chroma) -> Set[str]:
    ids: Set[str] = set()
    offset = 0
    limit = 5000
    while True:
        batch = (
            vectorstore._collection.get(  # pylint: disable=protected-access
                ids=None, where=None, limit=limit, offset=offset
            )
        )
        batch_ids = batch.get("ids", [])
        if not batch_ids:
            break
        ids.update(batch_ids)
        offset += len(batch_ids)
        if len(batch_ids) < limit:
            break
    return ids


def _load_json(path: str) -> Dict[str, str]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.debug("Loaded cache file %s with %d entries", path, len(data))
        return data
    logger.debug("No cache file found at %s", path)
    return {}


def _dump_json(path: str, data: Dict[str, str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.debug("Dumped %d entries to %s", len(data), path)
