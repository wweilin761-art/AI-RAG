import base64
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from PIL import Image
from typing_extensions import Annotated

from .config import Config
from .doc_processor import FullDocProcessor
from .memory_manager import MultimodalMemoryManager
from .rag_retriever import MultiPathRAGRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(Config.LOG_DIR, "langgraph_agent.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("LangGraphAgent")

_doc_processor: Optional[FullDocProcessor] = None
_rag_retriever: Optional[MultiPathRAGRetriever] = None
_memory_manager: Optional[MultimodalMemoryManager] = None


def get_doc_processor() -> FullDocProcessor:
    global _doc_processor
    if _doc_processor is None:
        _doc_processor = FullDocProcessor()
    return _doc_processor


def get_rag_retriever() -> MultiPathRAGRetriever:
    global _rag_retriever
    if _rag_retriever is None:
        _rag_retriever = MultiPathRAGRetriever()
    return _rag_retriever


def get_memory_manager() -> MultimodalMemoryManager:
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MultimodalMemoryManager()
    return _memory_manager


class AgentState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    question: str
    image_path: Optional[str]
    image_base64: Optional[str]
    image_description: Optional[str]
    input_type: Optional[str]
    intent: Optional[str]
    metadata_filters: Optional[Dict[str, Any]]
    retrieved_docs: List[Dict[str, Any]]
    scratchpad: str
    iterations: int
    doc_upload_path: Optional[str]
    doc_delete_name: Optional[str]


@tool
def rag_retrieve_tool(query: str) -> str:
    """Retrieve documents from the knowledge base for a query."""
    docs = get_rag_retriever().retrieve(query)
    return json.dumps(docs, ensure_ascii=False)


@tool
def doc_upload_tool(file_path: str) -> str:
    """Process a local file and add it into the knowledge base."""
    raw_docs = get_doc_processor().process_and_split(file_path)
    result = get_rag_retriever().add_documents(raw_docs)
    if os.path.exists(file_path) and "uploads" in file_path:
        os.remove(file_path)
    return result


@tool
def doc_delete_tool(source_filename: str) -> str:
    """Delete all indexed chunks for a source filename."""
    return get_rag_retriever().delete_documents(source_filename)


tools = [rag_retrieve_tool, doc_upload_tool, doc_delete_tool]
tool_node = ToolNode(tools)

INTENT_RECOGNITION_PROMPT = """
Classify the user request into exactly one intent:
- chat: general chat
- qa: knowledge base question answering
- doc_upload: user wants to upload or index a document
- doc_delete: user wants to delete indexed content

Only return one label from: chat, qa, doc_upload, doc_delete

User request:
{question}
"""

CHAT_PROMPT = """
You are a helpful AI assistant.

Conversation context:
{context}

User question:
{question}
"""

MULTIMODAL_UNDERSTANDING_PROMPT = """
Carefully describe the image in under 300 words, focusing on:
1. The main subject
2. Important objects, text, and scene details
3. Any chart, table, or screen information
"""

TEXT_QA_REACT_PROMPT = """
You are a professional RAG assistant. Use tools when helpful.

Conversation context:
{context}

User question:
{question}

Retrieved documents:
{retrieved_docs_str}

Iteration:
{iterations}/{max_iterations}
"""

MULTIMODAL_QA_REACT_PROMPT = """
You are a professional multimodal RAG assistant. Use tools when helpful.

Conversation context:
{context}

Image description:
{image_description}

User question:
{question}

Retrieved documents:
{retrieved_docs_str}

Iteration:
{iterations}/{max_iterations}
"""

DOC_MANAGEMENT_PROMPT = """
You are a document management assistant. Help the user manage indexed files.

Intent:
{intent}

Conversation context:
{context}

User request:
{question}
"""


def _safe_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _docs_to_prompt(docs: List[Dict[str, Any]], limit: int) -> str:
    if not docs:
        return "No retrieved documents yet."

    lines = []
    for index, doc in enumerate(docs[:limit], start=1):
        lines.append(
            "\n".join(
                [
                    f"[Document {index}]",
                    f"Source: {doc.get('metadata', {}).get('source', 'unknown')}",
                    f"Page: {doc.get('metadata', {}).get('page', 'unknown')}",
                    f"Score: {doc.get('rerank_score', 'unknown')}",
                    f"Content: {doc.get('content', '')}",
                ]
            )
        )
    return "\n\n".join(lines)


def create_multimodal_rag_langgraph():
    llm = ChatOpenAI(
        model=Config.LLM_MODEL_NAME,
        temperature=Config.LLM_TEMPERATURE,
        max_tokens=Config.LLM_MAX_TOKENS,
        api_key=Config.OPENAI_API_KEY,
        base_url=Config.OPENAI_API_BASE,
    )
    llm_with_tools = llm.bind_tools(tools)
    multimodal_llm = ChatOpenAI(
        model=Config.MULTIMODAL_MODEL_NAME,
        temperature=Config.LLM_TEMPERATURE,
        max_tokens=Config.LLM_MAX_TOKENS,
        api_key=Config.OPENAI_API_KEY,
        base_url=Config.OPENAI_API_BASE,
    )

    max_iterations = Config.MAX_REACT_ITERATIONS
    max_docs_in_context = Config.MAX_DOCS_IN_CONTEXT

    def input_preprocessing_node(state: AgentState) -> AgentState:
        image_path = state.get("image_path")
        image_base64 = None
        input_type = "text_only"

        if image_path and os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")
            image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            resized_path = os.path.join(
                Config.IMAGE_DIR,
                f"resized_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.path.basename(image_path)}",
            )
            image.save(resized_path)
            with open(resized_path, "rb") as file:
                image_base64 = base64.b64encode(file.read()).decode("utf-8")
            input_type = "text_with_image"

        return {
            "image_path": image_path,
            "image_base64": image_base64,
            "input_type": input_type,
            "retrieved_docs": state.get("retrieved_docs", []),
            "iterations": state.get("iterations", 0),
            "scratchpad": f"Preprocessed input: {input_type}",
        }

    def multimodal_understanding_node(state: AgentState) -> AgentState:
        image_base64 = state.get("image_base64")
        if not image_base64:
            return {"image_description": "", "scratchpad": "No image provided"}

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": MULTIMODAL_UNDERSTANDING_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                ],
            }
        ]

        try:
            response = multimodal_llm.invoke(messages)
            description = _safe_content(response.content).strip()
        except Exception as exc:
            logger.warning("Multimodal understanding failed: %s", exc)
            description = "Image understanding failed."

        return {"image_description": description, "scratchpad": "Image described"}

    def intent_recognition_node(state: AgentState) -> AgentState:
        if state.get("input_type") == "text_with_image":
            return {"intent": "qa", "iterations": 0, "scratchpad": "Image input defaults to QA"}

        prompt = INTENT_RECOGNITION_PROMPT.format(question=state["question"])
        try:
            response = llm.invoke(
                [
                    SystemMessage(content="You classify user intent."),
                    HumanMessage(content=prompt),
                ]
            )
            intent = _safe_content(response.content).strip().lower()
        except Exception as exc:
            logger.warning("Intent recognition failed: %s", exc)
            intent = "qa"

        if intent not in {"chat", "qa", "doc_upload", "doc_delete"}:
            intent = "qa"
        return {"intent": intent, "iterations": 0, "scratchpad": f"Intent: {intent}"}

    def chat_node(state: AgentState) -> AgentState:
        memory_manager = get_memory_manager()
        prompt = CHAT_PROMPT.format(
            context=memory_manager.get_context(),
            question=state["question"],
        )
        try:
            response = llm.invoke(
                [
                    SystemMessage(content="You are a helpful assistant."),
                    HumanMessage(content=prompt),
                ]
            )
            answer = _safe_content(response.content)
        except Exception as exc:
            logger.warning("Chat generation failed: %s", exc)
            answer = (
                "Sorry, I could not reach the LLM service. "
                "Please check API key, base URL, and network connectivity."
            )

        memory_manager.add_user_message(state["question"])
        memory_manager.add_ai_message(answer)
        return {"messages": [AIMessage(content=answer)], "scratchpad": "Chat answered"}

    def multimodal_qa_react_node(state: AgentState) -> AgentState:
        memory_manager = get_memory_manager()
        prompt = MULTIMODAL_QA_REACT_PROMPT.format(
            context=memory_manager.get_context(),
            image_description=state.get("image_description", ""),
            question=state["question"],
            retrieved_docs_str=_docs_to_prompt(state.get("retrieved_docs", []), max_docs_in_context),
            iterations=state.get("iterations", 0),
            max_iterations=max_iterations,
        )
        try:
            response = llm_with_tools.invoke(
                [
                    SystemMessage(content="You are a multimodal RAG assistant."),
                    HumanMessage(content=prompt),
                ]
            )
        except Exception as exc:
            logger.warning("Multimodal QA failed: %s", exc)
            response = AIMessage(
                content=(
                    "Sorry, I could not reach the LLM service. "
                    "Please check API key, base URL, and network connectivity."
                )
            )

        return {
            "messages": [response],
            "iterations": state.get("iterations", 0) + 1,
            "scratchpad": "Multimodal QA step completed",
        }

    def text_qa_react_node(state: AgentState) -> AgentState:
        memory_manager = get_memory_manager()
        prompt = TEXT_QA_REACT_PROMPT.format(
            context=memory_manager.get_context(),
            question=state["question"],
            retrieved_docs_str=_docs_to_prompt(state.get("retrieved_docs", []), max_docs_in_context),
            iterations=state.get("iterations", 0),
            max_iterations=max_iterations,
        )
        try:
            response = llm_with_tools.invoke(
                [
                    SystemMessage(content="You are a RAG assistant."),
                    HumanMessage(content=prompt),
                ]
            )
        except Exception as exc:
            logger.warning("Text QA failed: %s", exc)
            response = AIMessage(
                content=(
                    "Sorry, I could not reach the LLM service. "
                    "Please check API key, base URL, and network connectivity."
                )
            )

        return {
            "messages": [response],
            "iterations": state.get("iterations", 0) + 1,
            "scratchpad": "Text QA step completed",
        }

    def tool_result_node(state: AgentState) -> AgentState:
        messages = state.get("messages", [])
        if not messages:
            return {}

        last_message = messages[-1]
        if not isinstance(last_message, ToolMessage):
            return {}

        retrieved_docs = state.get("retrieved_docs", [])
        if last_message.name == "rag_retrieve_tool":
            try:
                parsed = json.loads(_safe_content(last_message.content))
                if isinstance(parsed, list):
                    retrieved_docs = parsed
            except Exception as exc:
                logger.warning("Failed to parse rag tool output: %s", exc)

        return {
            "retrieved_docs": retrieved_docs,
            "scratchpad": f"Processed tool result from {last_message.name}",
        }

    def doc_management_node(state: AgentState) -> AgentState:
        memory_manager = get_memory_manager()
        prompt = DOC_MANAGEMENT_PROMPT.format(
            intent=state.get("intent", "qa"),
            context=memory_manager.get_context(),
            question=state["question"],
        )
        try:
            response = llm_with_tools.invoke(
                [
                    SystemMessage(content="You help manage indexed documents."),
                    HumanMessage(content=prompt),
                ]
            )
        except Exception as exc:
            logger.warning("Document management failed: %s", exc)
            response = AIMessage(
                content=(
                    "Sorry, I could not reach the LLM service. "
                    "Please check API key, base URL, and network connectivity."
                )
            )

        return {"messages": [response], "scratchpad": "Document management step completed"}

    def final_answer_node(state: AgentState) -> AgentState:
        memory_manager = get_memory_manager()
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None

        final_answer = (
            _safe_content(last_message.content)
            if isinstance(last_message, AIMessage)
            else "Sorry, no final answer was generated."
        )

        if state.get("input_type") == "text_with_image":
            memory_manager.add_user_message(
                state["question"],
                state.get("image_description"),
                state.get("image_path"),
            )
        else:
            memory_manager.add_user_message(state["question"])
        memory_manager.add_ai_message(final_answer)

        return {"messages": [AIMessage(content=final_answer)], "scratchpad": "Final answer generated"}

    def input_type_branch(state: AgentState) -> str:
        return state.get("input_type", "text_only")

    def intent_branch(state: AgentState) -> str:
        return state.get("intent", "qa")

    def react_branch(state: AgentState) -> str:
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None

        if state.get("iterations", 0) >= max_iterations:
            return "final_answer"

        if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
            return "tools"

        return "final_answer"

    def doc_management_branch(state: AgentState) -> str:
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None
        if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
            return "tools"
        return "final_answer"

    def tool_result_return_branch(state: AgentState) -> str:
        if state.get("input_type") == "text_with_image":
            return "multimodal_qa_react"
        if state.get("intent") == "qa":
            return "text_qa_react"
        return "doc_management"

    workflow = StateGraph(AgentState)
    workflow.add_node("input_preprocessing", input_preprocessing_node)
    workflow.add_node("multimodal_understanding", multimodal_understanding_node)
    workflow.add_node("intent_recognition", intent_recognition_node)
    workflow.add_node("chat", chat_node)
    workflow.add_node("multimodal_qa_react", multimodal_qa_react_node)
    workflow.add_node("text_qa_react", text_qa_react_node)
    workflow.add_node("doc_management", doc_management_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("tool_result", tool_result_node)
    workflow.add_node("final_answer", final_answer_node)

    workflow.set_entry_point("input_preprocessing")
    workflow.add_conditional_edges(
        "input_preprocessing",
        input_type_branch,
        {
            "text_with_image": "multimodal_understanding",
            "text_only": "intent_recognition",
        },
    )
    workflow.add_edge("multimodal_understanding", "multimodal_qa_react")
    workflow.add_conditional_edges(
        "multimodal_qa_react",
        react_branch,
        {"tools": "tools", "final_answer": "final_answer"},
    )
    workflow.add_conditional_edges(
        "intent_recognition",
        intent_branch,
        {
            "chat": "chat",
            "qa": "text_qa_react",
            "doc_upload": "doc_management",
            "doc_delete": "doc_management",
        },
    )
    workflow.add_edge("chat", END)
    workflow.add_conditional_edges(
        "text_qa_react",
        react_branch,
        {"tools": "tools", "final_answer": "final_answer"},
    )
    workflow.add_conditional_edges(
        "doc_management",
        doc_management_branch,
        {"tools": "tools", "final_answer": "final_answer"},
    )
    workflow.add_edge("tools", "tool_result")
    workflow.add_conditional_edges(
        "tool_result",
        tool_result_return_branch,
        {
            "multimodal_qa_react": "multimodal_qa_react",
            "text_qa_react": "text_qa_react",
            "doc_management": "doc_management",
        },
    )
    workflow.add_edge("final_answer", END)
    return workflow.compile()
