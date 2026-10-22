"""Deployable Gradio app for the AI Study Assistant Agent."""

import os
import zipfile
from pathlib import Path

import gradio as gr
from google import genai
from google.genai import types

from module5_agent_tools import calculate, manage_tasks, search_notes

BASE_DIR = Path(__file__).resolve().parent
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

SYSTEM_INSTRUCTIONS = """You are a friendly AI Study Assistant Agent for a college student.
Explain concepts in simple, beginner-friendly language and use clear steps.
You can use tools when useful:
- calculate(expression): solve arithmetic accurately.
- search_notes(query): search the built-in study notes.
- manage_tasks(action, task): add, list, or complete study tasks.
When adding a task, use action='add' and include its description in task.
To list tasks, use action='list'. To complete a task, use action='complete' and
pass its task ID or exact task description. Do not claim a tool succeeded unless
the tool result confirms it. If notes search returns no match, say so clearly.
"""


def make_source_zip() -> str:
    """Create a safe source-code ZIP for visitors to download; never include secrets."""
    zip_path = BASE_DIR / "module5_ai_study_assistant_source.zip"
    include_names = ["app.py", "module5_agent_tools.py", "requirements.txt", "README.md"]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in include_names:
            file_path = BASE_DIR / name
            if file_path.is_file():
                archive.write(file_path, arcname=name)
    return str(zip_path)


SOURCE_ZIP = make_source_zip()


def make_prompt(message: str, memory: list[dict[str, str]]) -> str:
    recent = memory[-10:] if memory else []
    transcript = []
    for item in recent:
        role = "Student" if item.get("role") == "user" else "Assistant"
        content = item.get("content", "")
        transcript.append(f"{role}: {content}")
    transcript.append(f"Student: {message}")
    transcript.append("Assistant: Please answer the student's latest message.")
    return "Conversation so far:\n" + "\n".join(transcript)


def agent_chat(message: str, memory: list[dict[str, str]] | None):
    memory = memory or []
    message = (message or "").strip()
    if not message:
        display = "\n\n".join(
            f"{'You' if item['role'] == 'user' else 'Agent'}: {item['content']}"
            for item in memory
        )
        return display, memory, ""

    if not API_KEY:
        answer = (
            "Configuration needed: set the GEMINI_API_KEY environment variable "
            "in your deployment settings. Do not put your API key in GitHub code."
        )
    else:
        try:
            client = genai.Client(api_key=API_KEY)
            prompt = make_prompt(message, memory)
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[calculate, search_notes, manage_tasks],
                    system_instruction=SYSTEM_INSTRUCTIONS,
                    temperature=0.2,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=False
                    ),
                ),
            )
            answer = response.text or "I couldn't create a text answer. Please try again."
        except Exception as exc:
            # Keep the API key and environment values out of error messages.
            answer = (
                "Sorry, the agent could not complete that request. Check that "
                "GEMINI_API_KEY is set, the selected model is available to your "
                "API key, and the installed packages are up to date.\n\n"
                f"Technical detail: {type(exc).__name__}: {exc}"
            )

    updated_memory = memory + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]
    updated_memory = updated_memory[-20:]
    display = "\n\n".join(
        f"{'You' if item['role'] == 'user' else 'Agent'}: {item['content']}"
        for item in updated_memory
    )
    return display, updated_memory, ""


def clear_conversation():
    return "", [], ""


with gr.Blocks(title="AI Study Assistant Agent", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🤖 AI Study Assistant Agent\n"
        "Ask study questions, search saved notes, use a calculator, and manage study tasks."
    )
    gr.Markdown("## 📥 Download Project Code\nDownload the source files to explore how this AI agent works.")
    gr.DownloadButton(
        label="⬇️ Download Complete Project Code (ZIP)",
        value=SOURCE_ZIP,
        variant="secondary",
    )
    conversation_box = gr.Textbox(
        label="Conversation", lines=18, interactive=False, placeholder="Your conversation will appear here."
    )
    memory_state = gr.State([])
    message_box = gr.Textbox(
        label="Ask your study question",
        placeholder="Example: Explain Java Collections or calculate 125 * 8",
        lines=2,
    )
    with gr.Row():
        send_button = gr.Button("Send", variant="primary")
        clear_button = gr.Button("Clear Conversation")

    send_button.click(
        fn=agent_chat,
        inputs=[message_box, memory_state],
        outputs=[conversation_box, memory_state, message_box],
    )
    message_box.submit(
        fn=agent_chat,
        inputs=[message_box, memory_state],
        outputs=[conversation_box, memory_state, message_box],
    )
    clear_button.click(
        fn=clear_conversation,
        inputs=[],
        outputs=[conversation_box, memory_state, message_box],
    )


if __name__ == "__main__":
    # Render supplies PORT. The default keeps local testing straightforward.
    port = int(os.getenv("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port)
