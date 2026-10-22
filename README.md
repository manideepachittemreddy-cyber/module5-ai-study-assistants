# AI Study Assistant Agent — Module 5

A beginner-friendly Gradio application using the Gemini API and function-calling tools. The app can answer study questions, calculate basic arithmetic, search built-in study notes, and manage a small list of study tasks.

## Features

- Gemini-powered study assistant
- Calculator tool (safe arithmetic parser; does not use `eval`)
- Search tool for the built-in Java, Collections Framework, RAG, embeddings, and AI agent notes
- Add, list, and complete study tasks
- Conversation context during the current app session
- Download button so website visitors can download the project source ZIP

## Run locally

Use Python 3.10 or newer.

```bash
python -m venv .venv
```

Activate the environment, then run:

```bash
pip install -r requirements.txt
```

Set the Gemini API key as an environment variable. Do not paste a real key into the source code or commit it to GitHub.

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

Open the local URL printed in the terminal.

## Deploy on Render

1. Upload `app.py`, `module5_agent_tools.py`, `requirements.txt`, and `README.md` to a GitHub repository.
2. In Render, create a **Web Service** from that repository.
3. Set **Build Command** to `pip install -r requirements.txt`.
4. Set **Start Command** to `python app.py`.
5. In the Render service's Environment settings, add `GEMINI_API_KEY` with your own Gemini API key.
6. Optionally set `GEMINI_MODEL` if your API key uses a different available model. The default is `gemini-3.5-flash-lite`.
7. Deploy the service.

Never place your API key in GitHub, README files, screenshots, or the downloadable ZIP.

## Notes and limitations

- The built-in notes are sample notes. Edit `study_notes` in `module5_agent_tools.py` to add your own learning material.
- Tasks are stored in memory and reset when the app process restarts or redeploys. This project does not use a persistent database.
- Conversation state is per browser session and is not stored permanently.
- API usage can incur charges or have rate limits depending on your Google AI account and current plan.
