# AI Study Assistant Agent — GenAI Capstone

A Gradio application powered by the Google Gemini API. It demonstrates prompt engineering, tool/function calling, and retrieval-augmented generation (RAG) for text-based PDFs.

## Features
- Gemini-powered study assistant with recent conversation context
- Upload a PDF, extract text, split it into overlapping chunks, embed chunks, retrieve semantically relevant chunks, and generate answers grounded in retrieved context
- Calculator, built-in study notes search, and study task manager tools
- Download button for the source project

## Run locally
Use Python 3.10 or newer. In the project directory, install packages:

```bash
pip install -r requirements.txt
```

Set the API key as an environment variable, then run the app. Never commit your real API key to GitHub.

Windows PowerShell:
```powershell
$env:GEMINI_API_KEY="YOUR_API_KEY"
python app.py
```

macOS/Linux:
```bash
export GEMINI_API_KEY="YOUR_API_KEY"
python app.py
```

## Deploy on Render
1. Create a Render Web Service connected to this GitHub repository.
2. Build Command: `pip install -r requirements.txt`
3. Start Command: `python app.py`
4. In Environment settings, set `GEMINI_API_KEY` to your own Gemini API key.
5. Optional: set `GEMINI_MODEL` and `GEMINI_EMBEDDING_MODEL` only to model names available to your API key. Defaults are `gemini-3.5-flash-lite` and `gemini-embedding-001`.
6. Deploy, then test chat and PDF upload with a small text-based PDF.

## Limitations
- RAG index is kept in browser session memory and is lost when the session ends or the service restarts.
- Built-in notes are examples and can be edited in `module5_agent_tools.py`.
- Tasks are stored in memory and reset when the Python process restarts or redeploys.
- Scanned/image-only PDFs may not contain extractable text; OCR is not included.
- API access, model availability, quotas, and costs depend on the Google AI account.
