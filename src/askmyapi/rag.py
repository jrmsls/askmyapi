from langchain.chains.history_aware_retriever import (
    create_history_aware_retriever,
)
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor


def create_rag_chain(llm, retriever):
    """
    Build a history-aware, compression-enabled QA chain tailored for API docs.

    Important fix:
    - ContextualCompressionRetriever expects a *string* query, not a dict.
      Therefore we must wrap the base retriever with compression FIRST,
      and then make it history-aware.
    - Before, the order was reversed and `history_aware` was passed to
      ContextualCompressionRetriever, causing a type mismatch error.
    """

    # Chat memory (keeps track of conversation history)
    memory = ConversationBufferMemory(
        return_messages=True, memory_key="chat_history"
    )

    # --- Step 1: Add a compression layer on the base retriever
    compressor = LLMChainExtractor.from_llm(llm)
    compressed_base = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )

    # --- Step 2: Make the retriever history-aware
    history_aware = create_history_aware_retriever(
        llm=llm,
        retriever=compressed_base,
        prompt=ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an API documentation assistant. Expand the user's query "
                    "with synonyms for OpenAPI concepts (operationId, path, parameters, "
                    "requestBody, responses).",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
            ]
        ),
    )

    # --- Step 3: Build the QA prompt (always return method/path, params, curl example)
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer strictly using the provided API docs.\n"
                "Always include:\n"
                "1) Method & path\n"
                "2) Required params (name, in, type)\n"
                "3) Example curl (with base URL if present)\n"
                "If uncertain, say so and propose how to verify.\n"
                "Cite sources as (METHOD PATH/status) when relevant.",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("system", "Context:\n{context}"),
        ]
    )

    # --- Step 4: Assemble the final RAG chain
    document_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware, document_chain)

    return rag_chain, memory
