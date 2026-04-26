from typing import List, Dict

_llamaindex_embed_model = None


def get_embedding_model():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    except ImportError:
        print("sentence-transformers not installed. Install with: pip install sentence-transformers")
        return None
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        return None


def generate_embedding(text: str, model) -> List[float]:
    if model is None:
        return []
    try:
        return model.encode(text, convert_to_tensor=False).tolist()
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return []


def llamaindex_chunker(
    text: str,
    buffer_size: int = 1,
    threshold: int = 80,
    embed_model=None,
    debug: bool = False,
) -> tuple[List[Dict], Dict[str, int]]:
    global _llamaindex_embed_model

    try:
        from llama_index.core.node_parser import SemanticSplitterNodeParser
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.core import Document

        if _llamaindex_embed_model is None:
            _llamaindex_embed_model = HuggingFaceEmbedding(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

        splitter = SemanticSplitterNodeParser(
            buffer_size=buffer_size,
            breakpoint_percentile_threshold=threshold,
            embed_model=_llamaindex_embed_model,
        )

        nodes = splitter.get_nodes_from_documents([Document(text=text)])
        texts = [node.text for node in nodes]

        if embed_model and texts:
            embeddings = embed_model.encode(texts, convert_to_tensor=False).tolist()
        else:
            embeddings = [[] for _ in texts]

        chunks = [{'text': t, 'embedding': e} for t, e in zip(texts, embeddings)]

        empty_stats = {'small': 0, 'toc': 0, 'meaningless': 0, 'repetitive': 0, 'decorative': 0}
        return chunks, empty_stats

    except ImportError:
        print("LlamaIndex or HuggingFace embeddings not installed.")
        print("Install with: pip install llama-index llama-index-embeddings-huggingface sentence-transformers")
        return [], {}
    except Exception as e:
        print(f"LlamaIndex error: {e}")
        return [], {}
