import logging
import os
from typing import Any, Dict, List

import torch
from llama_index.core.schema import NodeWithScore
from llama_index.retrievers.bm25 import BM25Retriever
from sentence_transformers import CrossEncoder

from .config import Config
from .parent_child_index import ParentChildIndexManager, get_embed_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(Config.LOG_DIR, "rag_retriever.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("RAGRetriever")


class MultiPathRAGRetriever:
    def __init__(self) -> None:
        self.coarse_top_k_per_path = Config.COARSE_TOP_K_PER_PATH
        self.rerank_top_n = Config.RERANK_TOP_N
        get_embed_model()
        self.parent_child_index = ParentChildIndexManager()
        self.docstore = self.parent_child_index.docstore
        self._refresh_bm25_retriever()
        self.reranker = CrossEncoder(
            Config.RERANKER_MODEL_NAME,
            max_length=512,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    def _refresh_bm25_retriever(self) -> None:
        all_nodes = list(self.docstore.docs.values())
        parent_only_nodes = [
            node for node in all_nodes if "parent_id" not in getattr(node, "metadata", {})
        ]
        if parent_only_nodes:
            self.bm25_retriever = BM25Retriever.from_defaults(
                nodes=parent_only_nodes,
                similarity_top_k=self.coarse_top_k_per_path,
            )
        else:
            self.bm25_retriever = None

    def add_documents(self, raw_documents: List) -> str:
        result = self.parent_child_index.add_documents(raw_documents)
        self.docstore = self.parent_child_index.docstore
        self._refresh_bm25_retriever()
        return result

    def delete_documents(self, source_filename: str) -> str:
        result = self.parent_child_index.delete_documents(source_filename)
        self.docstore = self.parent_child_index.docstore
        self._refresh_bm25_retriever()
        return result

    def _multi_path_coarse_recall(self, query: str) -> List[NodeWithScore]:
        all_nodes: List[NodeWithScore] = []

        try:
            parent_child_retriever = self.parent_child_index.as_retriever(
                similarity_top_k=self.coarse_top_k_per_path,
                num_parents=10,
            )
            all_nodes.extend(parent_child_retriever.retrieve(query))
        except Exception as exc:
            logger.warning("Parent-child retrieval failed: %s", exc)

        if self.bm25_retriever is not None:
            try:
                all_nodes.extend(self.bm25_retriever.retrieve(query))
            except Exception as exc:
                logger.warning("BM25 retrieval failed: %s", exc)

        seen = set()
        unique_nodes = []
        for item in all_nodes:
            node_id = item.node.node_id
            if node_id in seen:
                continue
            seen.add(node_id)
            unique_nodes.append(item)
        return unique_nodes

    def _two_stage_rerank(self, query: str, coarse_nodes: List[NodeWithScore]) -> List[Dict[str, Any]]:
        if not coarse_nodes:
            return []

        pairs = [(query, node.node.text) for node in coarse_nodes]
        try:
            scores = self.reranker.predict(pairs, batch_size=8)
        except Exception as exc:
            logger.warning("Cross-encoder rerank failed, falling back to retriever scores: %s", exc)
            scores = [node.score or 0.0 for node in coarse_nodes]

        for node, score in zip(coarse_nodes, scores):
            node.score = float(score)

        reranked_nodes = sorted(
            coarse_nodes,
            key=lambda item: item.score or 0.0,
            reverse=True,
        )[: self.rerank_top_n]

        return [
            {
                "node_id": node.node.node_id,
                "content": node.node.text,
                "metadata": node.node.metadata,
                "rerank_score": round(node.score or 0.0, 4),
            }
            for node in reranked_nodes
        ]

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        coarse_nodes = self._multi_path_coarse_recall(query)
        return self._two_stage_rerank(query, coarse_nodes)
