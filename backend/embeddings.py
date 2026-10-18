import os
from functools import lru_cache
from typing import Callable, List

from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings


@lru_cache(maxsize=1)
def get_embedding_function() -> Callable[[str], List[float]]:
    """Return a callable that converts a text string into a vector.

    The implementation is chosen via the ``EMBEDDING_MODEL`` environment variable:
    * ``openai`` – uses OpenAI's ``text-embedding-3-small`` (or the model
      specified in ``OPENAI_EMBEDDING_MODEL``).
    * Any other value – interpreted as a HuggingFace model name and loaded
      with ``HuggingFaceEmbeddings``.
    The returned function matches the signature expected by the ingestion
    pipeline (a single string argument returning a list of floats).
    """
    model_cfg = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    if model_cfg.lower() == "openai":
        openai_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        embedder = OpenAIEmbeddings(model=openai_model)
    else:
        # Assume a HF model identifier; the library will download it on first use.
        embedder = HuggingFaceEmbeddings(model_name=model_cfg)

    # LangChain's embedder objects expose ``embed_query`` for a single string.
    return embedder.embed_query
