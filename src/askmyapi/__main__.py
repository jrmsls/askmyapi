import argparse
import re
import logging

from askmyapi.config import load_environment, get_llm
from askmyapi.spec_loader import load_and_deref_spec
from askmyapi.ingestion import openapi_to_documents
from askmyapi.vectorstore import setup_vectorstore
from askmyapi.rag import create_rag_chain
from askmyapi.interface import launch_chat_interface

logger = logging.getLogger(__name__)


def _slugify(title: str) -> str:
    """Create a filesystem- and collection-friendly slug from an API title."""
    slug = re.sub(r"\s+", "_", title.strip().lower())
    slug = re.sub(r"[^a-z0-9_]+", "", slug)
    return slug or "api"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AskMyAPI: Query an OpenAPI specification through a RAG chatbot"
    )
    parser.add_argument("spec_path", help="Path to the OpenAPI JSON/YAML file")
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip OpenAPI validation step",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    load_environment()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    llm = get_llm()
    logger.info("LLM selected: %s", llm.__class__.__name__)

    # Load and dereference the spec
    logger.info("Loading spec from: %s", args.spec_path)
    spec, spec_hash = load_and_deref_spec(
        args.spec_path, validate=not args.no_validate
    )
    api_name = _slugify(spec.get("info", {}).get("title", "api"))
    logger.info("Spec loaded. API name: %s, hash: %s", api_name, spec_hash)

    # Build documents and vector store
    logger.info("Converting spec to documents...")
    documents = openapi_to_documents(
        spec, api_name=api_name, spec_hash=spec_hash
    )
    logger.info("Generated %d documents", len(documents))

    logger.info("Setting up vectorstore...")
    retriever, docstore = setup_vectorstore(
        documents, llm, api_name=api_name, spec_hash=spec_hash
    )
    logger.info("Vectorstore ready")

    # Create the RAG chain and memory
    logger.info("Creating RAG chain...")
    rag_chain, memory = create_rag_chain(llm, retriever)
    logger.info("RAG chain created successfully")

    # Launch UI
    logger.info("Launching Gradio interface...")
    launch_chat_interface(
        rag_chain=rag_chain,
        memory=memory,
        retriever=retriever,
        docstore=docstore,
        llm=llm,
        api_name=api_name,
        spec_hash=spec_hash,
    )


if __name__ == "__main__":
    main()
