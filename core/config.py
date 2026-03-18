import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
    OPENAI_API_BASE = os.getenv(
        "OPENAI_API_BASE",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", DASHSCOPE_API_KEY)

    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-large-zh-v1.5")
    RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-large-zh-v1.5")
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen-max")
    MULTIMODAL_MODEL_NAME = os.getenv("MULTIMODAL_MODEL_NAME", "qwen-vl-max")
    MULTIMODAL_EMBEDDING_MODEL_TYPE = os.getenv(
        "MULTIMODAL_EMBEDDING_MODEL_TYPE",
        "open_clip",
    )
    MULTIMODAL_EMBEDDING_MODEL_NAME = os.getenv(
        "MULTIMODAL_EMBEDDING_MODEL_NAME",
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    )
    MULTIMODAL_EMBEDDING_DIM = int(os.getenv("MULTIMODAL_EMBEDDING_DIM", 1024))
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.1))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 4096))

    MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
    MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "rag_full_integration_v2")
    MILVUS_DIM = int(os.getenv("MILVUS_DIM", 1024))
    MILVUS_INDEX_TYPE = os.getenv("MILVUS_INDEX_TYPE", "HNSW")
    MILVUS_METRIC_TYPE = os.getenv("MILVUS_METRIC_TYPE", "IP")
    MILVUS_HNSW_M = int(os.getenv("MILVUS_HNSW_M", 16))
    MILVUS_HNSW_EF_CONSTRUCTION = int(os.getenv("MILVUS_HNSW_EF_CONSTRUCTION", 200))
    MILVUS_HNSW_EF_SEARCH = int(os.getenv("MILVUS_HNSW_EF_SEARCH", 128))

    COARSE_TOP_K_PER_PATH = int(os.getenv("COARSE_TOP_K_PER_PATH", 30))
    RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", 8))
    MAX_REACT_ITERATIONS = int(os.getenv("MAX_REACT_ITERATIONS", 3))
    MAX_DOCS_IN_CONTEXT = int(os.getenv("MAX_DOCS_IN_CONTEXT", 6))

    PARENT_CHUNK_SIZE = int(os.getenv("PARENT_CHUNK_SIZE", 2048))
    PARENT_CHUNK_OVERLAP = int(os.getenv("PARENT_CHUNK_OVERLAP", 256))
    CHILD_CHUNK_SIZE = int(os.getenv("CHILD_CHUNK_SIZE", 384))
    CHILD_CHUNK_OVERLAP = int(os.getenv("CHILD_CHUNK_OVERLAP", 64))
    OCR_LANG = os.getenv("OCR_LANG", "ch")

    MEMORY_MAX_TOKENS = int(os.getenv("MEMORY_MAX_TOKENS", 8000))
    MEMORY_SUMMARY_THRESHOLD = int(os.getenv("MEMORY_SUMMARY_THRESHOLD", 6000))
    MEMORY_LONG_TERM_THRESHOLD = int(os.getenv("MEMORY_LONG_TERM_THRESHOLD", 10))

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    DATA_DIR = os.path.join(BASE_DIR, "data")
    UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
    IMAGE_DIR = os.path.join(DATA_DIR, "images")
    PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

    for dir_path in [LOG_DIR, UPLOAD_DIR, IMAGE_DIR, PROCESSED_DIR]:
        os.makedirs(dir_path, exist_ok=True)
