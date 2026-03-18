import os

from langchain_core.messages import AIMessage

from core.langgraph_agent import create_multimodal_rag_langgraph


def main() -> None:
    print("=" * 80)
    print("Multimodal RAG QA System")
    print("=" * 80)

    app = create_multimodal_rag_langgraph()

    print("\nSystem started.")
    print("Type 'quit' or 'exit' to stop.")
    print("Type 'image: /path/to/image.jpg [question]' to test image input.")
    print("=" * 80)

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in {"quit", "exit"}:
                print("Bye.")
                break
            if not user_input:
                continue

            image_path = None
            question = user_input

            if user_input.lower().startswith("image:"):
                parts = user_input[6:].strip().split(maxsplit=1)
                if not parts:
                    print("AI: Please provide an image path.")
                    continue
                image_path = parts[0]
                question = parts[1] if len(parts) > 1 else "Please describe this image."
                if not os.path.exists(image_path):
                    print(f"AI: Image path does not exist: {image_path}")
                    continue

            initial_state = {
                "messages": [],
                "question": question,
                "image_path": image_path,
                "image_base64": None,
                "image_description": None,
                "input_type": None,
                "intent": None,
                "metadata_filters": None,
                "retrieved_docs": [],
                "scratchpad": "",
                "iterations": 0,
                "doc_upload_path": None,
                "doc_delete_name": None,
            }

            print("AI: Processing...", end="", flush=True)
            result = app.invoke(initial_state)

            final_message = result["messages"][-1]
            if isinstance(final_message, AIMessage):
                print(f"\rAI: {final_message.content}")
            else:
                print("\rAI: No final AI message returned.")

        except KeyboardInterrupt:
            print("\n\nBye.")
            break
        except Exception as exc:
            print(f"\rAI: System error: {exc}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
