import logging
import os
from typing import List

import torch
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser.text import SentenceSplitter
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore

from .config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(Config.LOG_DIR, "parent_child_index.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("ParentChildIndex")

Settings.embed_model = HuggingFaceEmbedding(
    model_name=Config.EMBEDDING_MODEL_NAME,
    max_length=512,
    embed_batch_size=10,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

SPLITTER_SEPARATORS = ["\n", "。", "！", "？", "；", " ", ""]


class ParentChildRetriever(BaseRetriever):
    def __init__(
        self,
        vector_retriever: BaseRetriever,
        docstore: SimpleDocumentStore,
        parent_id_field: str = "parent_id",
        num_parents: int = 5,
    ) -> None:
        self.vector_retriever = vector_retriever
        self.docstore = docstore
        self.parent_id_field = parent_id_field
        self.num_parents = num_parents
        super().__init__()

    def _retrieve(self, query_bundle):
        child_nodes = self.vector_retriever.retrieve(query_bundle)
        parent_ids = set()
        parent_nodes: List[NodeWithScore] = []

        for child_node in child_nodes:
            parent_id = child_node.node.metadata.get(self.parent_id_field)
            if not parent_id or parent_id in parent_ids:
                continue
            parent_ids.add(parent_id)
            try:
                parent_node = self.docstore.get_node(parent_id)
                parent_nodes.append(NodeWithScore(node=parent_node, score=child_node.score))
            except Exception as exc:
                logger.warning("Parent node not found for %s: %s", parent_id, exc)

        parent_nodes.sort(key=lambda item: item.score or 0.0, reverse=True)
        return parent_nodes[: self.num_parents]


class ParentChildIndexManager:
    def __init__(self) -> None:
        self.parent_splitter = SentenceSplitter(
            chunk_size=Config.PARENT_CHUNK_SIZE,
            chunk_overlap=Config.PARENT_CHUNK_OVERLAP,
            separator="\n\n",
            secondary_separators=SPLITTER_SEPARATORS,
            paragraph_separator="\n\n\n",
        )
        self.child_splitter = SentenceSplitter(
            chunk_size=Config.CHILD_CHUNK_SIZE,
            chunk_overlap=Config.CHILD_CHUNK_OVERLAP,
            separator="\n\n",
            secondary_separators=SPLITTER_SEPARATORS,
            paragraph_separator="\n\n\n",
        )

        self.vector_store = MilvusVectorStore(
            host=Config.MILVUS_HOST,
            port=Config.MILVUS_PORT,
            collection_name=Config.MILVUS_COLLECTION_NAME,
            dim=Config.MILVUS_DIM,
            index_config={
                "index_type": Config.MILVUS_INDEX_TYPE,
                "metric_type": Config.MILVUS_METRIC_TYPE,
                "params": {
                    "M": Config.MILVUS_HNSW_M,
                    "efConstruction": Config.MILVUS_HNSW_EF_CONSTRUCTION,
                },
            },
            search_config={"params": {"ef": Config.MILVUS_HNSW_EF_SEARCH}},
            overwrite=False,
        )

        self.docstore = SimpleDocumentStore()
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store,
            docstore=self.docstore,
        )
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            storage_context=self.storage_context,
            embed_model=Settings.embed_model,
        )
        self.docstore = self.index.docstore

    def add_documents(self, raw_documents: List[Document]) -> str:
        all_parent_nodes = []
        all_child_nodes = []

        for raw_doc in raw_documents:
            parent_nodes = self.parent_splitter.get_nodes_from_documents([raw_doc])
            for parent_node in parent_nodes:
                child_nodes = self.child_splitter.get_nodes_from_documents([parent_node])
                for child_node in child_nodes:
                    child_node.metadata["parent_id"] = parent_node.node_id
                    child_node.metadata.update(parent_node.metadata)
                all_parent_nodes.append(parent_node)
                all_child_nodes.extend(child_nodes)

        if not all_parent_nodes:
            return "No valid content was extracted from the document."

        self.docstore.add_documents(all_parent_nodes)
        if all_child_nodes:
            self.index.insert_nodes(all_child_nodes)
        self.docstore = self.index.docstore
        return (
            f"Added {len(all_parent_nodes)} parent chunks and "
            f"{len(all_child_nodes)} child chunks."
        )

    def as_retriever(self, similarity_top_k: int = 30, num_parents: int = 5) -> ParentChildRetriever:
        vector_retriever = self.index.as_retriever(similarity_top_k=similarity_top_k)
        return ParentChildRetriever(
            vector_retriever=vector_retriever,
            docstore=self.docstore,
            parent_id_field="parent_id",
            num_parents=num_parents,
        )

    def delete_documents(self, source_filename: str) -> str:
        all_nodes = list(self.docstore.docs.values())
        parent_nodes = [
            node
            for node in all_nodes
            if node.metadata.get("source") == source_filename
            and "parent_id" not in node.metadata
        ]
        if not parent_nodes:
            return f"No documents found for source file: {source_filename}"

        parent_ids = [node.node_id for node in parent_nodes]
        child_nodes = [
            node for node in all_nodes if node.metadata.get("parent_id") in parent_ids
        ]
        child_ids = [node.node_id for node in child_nodes]

        if child_ids:
            try:
                self.index.delete_nodes(child_ids)
            except Exception as exc:
                logger.warning("Failed to delete child nodes from vector index: %s", exc)

        for parent_id in parent_ids:
            try:
                self.docstore.delete_document(parent_id)
            except Exception as exc:
                logger.warning("Failed to delete parent node %s: %s", parent_id, exc)

        self.docstore = self.index.docstore
        return (
            f"Deleted {len(parent_ids)} parent chunks and "
            f"{len(child_ids)} child chunks for {source_filename}."
        )
