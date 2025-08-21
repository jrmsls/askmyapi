import os
import uuid
import logging
from typing import Optional, List

import gradio as gr
from langchain.schema import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever

from askmyapi.vectorstore import setup_vectorstore

logger = logging.getLogger(__name__)


def _safe_content(msg_or_str) -> str:
    """Return a string content from a ChatMessage or raw string."""
    return getattr(msg_or_str, "content", msg_or_str)


def launch_chat_interface(
    rag_chain,
    memory,
    retriever: MultiVectorRetriever,
    docstore,
    llm,
    *,
    api_name: str,
    spec_hash: str,
    server_name: Optional[str] = None,
    server_port: Optional[int] = None,
    share: Optional[bool] = None,
    max_learn_chars: int = 50_000,
) -> None:
    """
    Launch the Gradio chat UI.
    """

    def _truncate_if_needed(text: str) -> str:
        if len(text) > max_learn_chars:
            return text[:max_learn_chars]
        return text

    def add_text_to_vectorstore(text: str, source: str = "user") -> str:
        text = _truncate_if_needed(text)
        if not text.strip():
            return "No content to index."

        doc = Document(
            page_content=text, metadata={"source": source, "kind": "note"}
        )
        try:
            setup_vectorstore(
                [doc], llm, api_name=api_name, spec_hash=spec_hash
            )
            return "Content indexed successfully."
        except Exception as e:
            logger.exception("Indexing failed")
            return f"Indexing failed: {e}"

    def handle_chat_input(
        user_input: str, history: Optional[List] = None
    ) -> str:
        """Main chat handler for Gradio ChatInterface."""
        _ = history or []
        try:
            logger.info("User input: %s", user_input)

            memory.chat_memory.add_user_message(user_input)
            response = rag_chain.invoke(
                {
                    "input": user_input,
                    "chat_history": memory.chat_memory.messages,
                }
            )

            logger.debug("Raw response type: %s", type(response))
            logger.debug("Raw response value: %s", response)

            if isinstance(response, dict):
                answer = (
                    response.get("answer")
                    or response.get("result")
                    or response.get("output_text")
                    or ""
                )
            else:
                answer = str(response) if response is not None else ""

            logger.info("Final answer: %s", answer)

            memory.chat_memory.add_ai_message(answer or "")
            return answer or "No answer produced."
        except Exception:
            logger.exception("Error in handle_chat_input")
            return "Unexpected error during chat handling."

    # --- Gradio UI
    with gr.Blocks(title="AskMyAPI - OpenAPI Chatbot") as app:
        gr.Markdown("# AskMyAPI")
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(height=500)
                msg = gr.Textbox(
                    label="Message", placeholder="Ask about your API…"
                )
                send = gr.Button("Send")

            with gr.Column(scale=1):
                gr.Markdown("### Extend the index")
                file_in = gr.File(
                    label="Upload .txt or .md to index",
                    file_types=[".txt", ".md"],
                    file_count="multiple",
                )
                txt_in = gr.Textbox(label="Paste text to index", lines=4)
                btn_index_files = gr.Button("Index uploaded files")
                btn_index_text = gr.Button("Index pasted text")
                index_status = gr.Textbox(
                    label="Indexing status",
                    lines=2,
                    interactive=False,
                )

        def _on_send(user_text, chat_history):
            if not user_text:
                return chat_history, ""
            answer = handle_chat_input(user_text, chat_history)
            chat_history = chat_history + [[user_text, answer]]
            return chat_history, ""

        def _on_index_files(files):
            if not files:
                return "No files selected."
            added = 0
            for f in files:
                try:
                    with open(f.name, "r", encoding="utf-8") as fp:
                        content = fp.read()
                except Exception as e:
                    logger.exception("Failed to read uploaded file %s", f.name)
                    return f"Failed to read '{os.path.basename(f.name)}': {e}"
                res = add_text_to_vectorstore(
                    content, source=os.path.basename(f.name)
                )
                if "successfully" in res:
                    added += 1
            return f"Indexed {added} file(s)."

        def _on_index_text(text):
            if not text or not text.strip():
                return "Nothing to index."
            return add_text_to_vectorstore(text, source="pasted_text")

        send.click(_on_send, [msg, chatbot], [chatbot, msg])
        btn_index_files.click(_on_index_files, [file_in], [index_status])
        btn_index_text.click(_on_index_text, [txt_in], [index_status])

    app.queue().launch(
        server_name=server_name
        or os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(
            os.getenv("GRADIO_SERVER_PORT", str(server_port or 7860))
        ),
        share=(
            share
            if share is not None
            else bool(int(os.getenv("GRADIO_SHARE", "0")))
        ),
    )
