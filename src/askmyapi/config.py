import os
import logging
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI


def load_environment() -> None:
    """Load .env and configure logging early in the program lifecycle."""
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")


def get_llm():
    """
    Return an LLM instance. Prefer OpenAI if a key is provided, otherwise fall back to Ollama.

    Note: LangChain's ChatOpenAI.invoke() returns a BaseMessage with a .content string.
    We normalize this downstream so callers can safely handle both ChatOpenAI and Ollama.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3")

    if openai_key:
        return ChatOpenAI(
            model=openai_model, temperature=0, api_key=openai_key
        )

    logging.warning("OpenAI API key not found. Using Ollama as fallback.")
    return OllamaLLM(model=ollama_model)
