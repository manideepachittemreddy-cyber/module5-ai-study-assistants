"""AI Study Assistant Agent with Gemini, tools, and PDF-based RAG."""

import os
import zipfile
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
from google import genai
from google.genai import types
from pypdf import PdfReader

from module5_agent_tools import calculate, manage_tasks, search_notes

BASE_DIR = Path(__file__).resolve().parent
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

SYSTEM_INSTRUCTIONS = """You are a friendly AI Study Assistant Agent for a college student.
Use simple, beginner-friendly language and clear steps. You may use tools:
- calculate(expression): perform arithmetic.
- search_notes(query): search the built-in study notes.
- manage_tasks(action, task): add, list, or complete study tasks.
When relevant PDF context is supplied, prioritize it for questions about the uploaded document.
Cite PDF page numbers when provided in the retrieved context. If the context does not answer
an upload-specific question, say that the uploaded PDF context does not contain enough detail.
Do not claim a tool succeeded unless its result confirms it.
"""


def get_client() -> genai.Client:
    if not API_KEY:
        raise RuntimeError("Set GEMINI_API_KEY in your environment or Render settings.")
    return genai.Client(api_key=API_KEY)


def split_pdf(file_path: str, chunk_size: int = 1400, overlap: int = 200) -> list[dict[str, Any]]:
    """Extract PDF text page-by-page and split it into overlapping chunks."""
    reader = PdfReader(file_path)
    chunks: list[dict[str, Any]] = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        start = 0
        while start < len(text):
            piece = text[start:start + chunk_size].strip()
            if piece:
                chunks.append({"text": piece, "page": page_num})
            if start + chunk_size >= len(text):
                break
            start += max(1, chunk_size - overlap)
    return chunks


def embed_texts(texts: list[str], task_type: str) -> list[list[float]]:
    """Generate Gemini embeddings in small batches."""
    client = get_client()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), 16):
        batch = texts[start:start + 16]
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        if not response.embeddings:
            raise RuntimeError("The embedding API returned no embeddings. Please try again.")
        vectors.extend([list(item.values) for item in response.embeddings])
    if len(vectors) != len(texts):
        raise RuntimeError("Embedding count did not match the number of PDF chunks.")
    return vectors


def process_pdf(file_path: str | None):
    if not file_path:
        return "Please choose a PDF file first.", []
    try:
        chunks = split_pdf(file_path)
        if not chunks:
            return "No readable text was found in this PDF. If it is a scanned PDF, run OCR first.", []
        vectors = embed_texts([item["text"] for item in chunks], "RETRIEVAL_DOCUMENT")
        for item, vector in zip(chunks, vectors):
            item["embedding"] = vector
        name = Path(file_path).name
        return f"✅ Processed {name}: indexed {len(chunks)} text chunks. Ask a question below.", chunks
    except Exception as exc:
        return f"Could not process the PDF ({type(exc).__name__}). Check your API key, model access, and PDF file, then try again.", []


def retrieve_context(question: str, rag_index: list[dict[str, Any]] | None, top_k: int = 4) -> str:
    if not rag_index:
        return ""
    query_vector = embed_texts([question], "RETRIEVAL_QUERY")[0]
    q = np.asarray(query_vector, dtype=float)
    q_norm = np.linalg.norm(q) or 1.0
    scored = []
    for item in rag_index:
        v = np.asarray(item["embedding"], dtype=float)
        denom = (np.linalg.norm(v) or 1.0) * q_norm
        score = float(np.dot(q, v) / denom)
        scored.append((score, item))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return "\n\n".join(
        f"[PDF page {item['page']}, relevance {score:.3f}]\n{item['text']}"
        for score, item in scored[:top_k]
    )


def make_prompt(message: str, memory: list[dict[str, str]], retrieved_context: str = "") -> str:
    transcript = []
    for item in (memory or [])[-10:]:
        role = "Student" if item.get("role") == "user" else "Assistant"
        transcript.append(f"{role}: {item.get('content', '')}")
    if retrieved_context:
        transcript.append(
            "Retrieved context from the student's uploaded PDF (use for document questions; cite page numbers):\n"
            + retrieved_context
        )
    transcript.append(f"Student: {message}")
    transcript.append("Assistant: Please answer the student's latest message.")
    return "Conversation so far:\n" + "\n\n".join(transcript)


def agent_chat(message: str, memory: list[dict[str, str]] | None, rag_index: list[dict[str, Any]] | None):
    memory = memory or []
    message = (message or "").strip()
    if not message:
        display = "\n\n".join(
            f"{'You' if item['role'] == 'user' else 'Agent'}: {item['content']}" for item in memory
        )
        return display, memory, ""

    try:
        client = get_client()
        context = retrieve_context(message, rag_index) if rag_index else ""
        prompt = make_prompt(message, memory, context)
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[calculate, search_notes, manage_tasks],
                system_instruction=SYSTEM_INSTRUCTIONS,
                temperature=0.2,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=False),
            ),
        )
        answer = response.text or "I couldn't create a text answer. Please try again."
    except Exception as exc:
        answer = (
            f"Sorry, I couldn't complete that request ({type(exc).__name__}). Check GEMINI_API_KEY, "
            "the selected Gemini models, and your API access, then try again."
        )

    updated_memory = (memory + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ])[-20:]
    display = "\n\n".join(
        f"{'You' if item['role'] == 'user' else 'Agent'}: {item['content']}" for item in updated_memory
    )
    return display, updated_memory, ""


def clear_conversation():
    return "", [], ""


def make_source_zip() -> str:
    """Package source files only; never include environment files or secrets."""
    zip_path = BASE_DIR / "module5_ai_study_assistant_source.zip"
    include_names = ["app.py", "module5_agent_tools.py", "requirements.txt", "README.md"]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in include_names:
            path = BASE_DIR / name
            if path.is_file():
                archive.write(path, arcname=name)
    return str(zip_path)


SOURCE_ZIP = make_source_zip()

with gr.Blocks(title="AI Study Assistant Agent", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🤖 AI Study Assistant Agent\n"
        "Ask study questions, upload a PDF for document-based answers, calculate values, and manage study tasks."
    )
    gr.Markdown("## 📄 1. Upload and process a PDF")
    pdf_file = gr.File(label="Choose a PDF document", file_types=[".pdf"], type="filepath")
    process_button = gr.Button("Process PDF", variant="secondary")
    pdf_status = gr.Markdown("No PDF processed yet.")
    rag_state = gr.State([])
    process_button.click(fn=process_pdf, inputs=[pdf_file], outputs=[pdf_status, rag_state])

    gr.Markdown("## 💬 2. Ask your study assistant")
    gr.Markdown("The agent can answer general questions, retrieve information from the processed PDF, use the calculator, search notes, and manage tasks.")
    gr.Markdown("## 📥 Download Project Code")
    gr.DownloadButton(label="⬇️ Download Complete Project Code (ZIP)", value=SOURCE_ZIP, variant="secondary")
    conversation_box = gr.Textbox(label="Conversation", lines=18, interactive=False, placeholder="Your conversation will appear here.")
    memory_state = gr.State([])
    message_box = gr.Textbox(label="Ask your study question", placeholder="Example: Summarize this PDF or calculate 125 * 8", lines=2)
    with gr.Row():
        send_button = gr.Button("Send", variant="primary")
        clear_button = gr.Button("Clear Conversation")
    send_button.click(fn=agent_chat, inputs=[message_box, memory_state, rag_state], outputs=[conversation_box, memory_state, message_box])
    message_box.submit(fn=agent_chat, inputs=[message_box, memory_state, rag_state], outputs=[conversation_box, memory_state, message_box])
    clear_button.click(fn=clear_conversation, inputs=[], outputs=[conversation_box, memory_state, message_box])


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port)
