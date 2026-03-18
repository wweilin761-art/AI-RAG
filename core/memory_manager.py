import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

import tiktoken
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(Config.LOG_DIR, "memory_manager.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("MemoryManager")


class MultimodalMemoryManager:
    def __init__(self) -> None:
        self.memory_max_tokens = Config.MEMORY_MAX_TOKENS
        self.memory_summary_threshold = Config.MEMORY_SUMMARY_THRESHOLD
        self.memory_long_term_threshold = Config.MEMORY_LONG_TERM_THRESHOLD
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.messages: List[BaseMessage] = []
        self.long_term_summary = ""
        self.image_descriptions: List[Dict[str, str]] = []

        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL_NAME,
            temperature=0.0,
            max_tokens=1024,
            api_key=Config.OPENAI_API_KEY,
            base_url=Config.OPENAI_API_BASE,
        )

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def _count_message_tokens(self, messages: List[BaseMessage]) -> int:
        total = 0
        for msg in messages:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            total += self._count_tokens(content)
        return total

    def _generate_summary(self, messages: List[BaseMessage], existing_summary: str = "") -> str:
        logger.info("Generating conversation summary")
        history = "\n".join(
            [
                f"{'User' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}"
                for msg in messages
            ]
        )
        prompt = f"""
Summarize the conversation below into a concise memory for future turns.

Existing summary:
{existing_summary or "None"}

Recent messages:
{history}
"""
        try:
            response = self.llm.invoke(
                [
                    SystemMessage(content="You summarize chat history for a multimodal assistant."),
                    HumanMessage(content=prompt),
                ]
            )
            return response.content.strip()
        except Exception as exc:
            logger.warning("Summary generation failed: %s", exc)
            return existing_summary

    def add_user_message(
        self,
        content: str,
        image_description: Optional[str] = None,
        image_path: Optional[str] = None,
    ) -> None:
        full_content = content
        if image_description:
            full_content = f"[User sent an image: {image_description}]\n{content}"
            self.image_descriptions.append(
                {
                    "timestamp": str(datetime.now()),
                    "description": image_description,
                    "path": image_path or "",
                }
            )
        self.messages.append(HumanMessage(content=full_content))
        self._check_and_update_summary()

    def add_ai_message(self, content: str) -> None:
        self.messages.append(AIMessage(content=content))
        self._check_and_update_summary()

    def _check_and_update_summary(self) -> None:
        total_tokens = self._count_message_tokens(self.messages)
        logger.info(
            "Current short-term memory tokens: %s/%s",
            total_tokens,
            self.memory_summary_threshold,
        )
        if total_tokens < self.memory_summary_threshold:
            return

        self.long_term_summary = self._generate_summary(self.messages, self.long_term_summary)
        keep = max(4, self.memory_long_term_threshold)
        self.messages = self.messages[-keep:]

    def get_context(self) -> str:
        parts: List[str] = []
        if self.long_term_summary:
            parts.append(f"[Summary]\n{self.long_term_summary}")

        if self.messages:
            parts.append("[Recent Conversation]")
            for msg in self.messages[-self.memory_long_term_threshold :]:
                role = "User" if isinstance(msg, HumanMessage) else "AI"
                parts.append(f"{role}: {msg.content}")

        if self.image_descriptions:
            parts.append("[Recent Images]")
            for i, item in enumerate(self.image_descriptions[-3:], start=1):
                parts.append(f"Image {i} ({item['timestamp']}): {item['description']}")

        return "\n".join(parts)

    def clear(self) -> None:
        self.messages = []
        self.long_term_summary = ""
        self.image_descriptions = []
