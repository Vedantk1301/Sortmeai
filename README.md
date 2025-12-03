# Sortme AI

Premium, production-grade fashion discovery experience with a Next.js front-end and a FastAPI backend that uses the OpenAI **Responses API** for both text search and image-based profile analysis.

## What's inside
- **Next.js UI (app router)** with a ChatGPT/Perplexity-inspired chat surface, responsive styling, and profile upload CTA.
- **Python FastAPI backend** that calls the OpenAI Responses API for text insights and portrait analysis (gender, skin tone, palette, fit notes, priorities).
- **requirements.txt** for backend dependencies plus CORS enabled for local development.

## Prerequisites
- Node.js 18.18+
- Python 3.10+
- An OpenAI API key with access to the Responses API (set as `OPENAI_API_KEY`)

## Backend setup (FastAPI + Responses API)
```bash
cd backend
python -m venv .venv
.\.venv\Scripts\activate   # or `source .venv/bin/activate` on macOS/Linux
pip install -r ../requirements.txt

# Environment (preferred: .env file in backend/)
echo OPENAI_API_KEY=sk-... > .env
echo OPENAI_MODEL=gpt-4.1-mini >> .env

# Or export in shell/CI
set OPENAI_API_KEY=sk-...         # or export OPENAI_API_KEY=...
set OPENAI_MODEL=gpt-4.1-mini

# Run the API
uvicorn main:app --reload --port 8000
```

Endpoints:
- `POST /api/analyze/text` - body: `{ "query": "Modern monochrome for a rooftop dinner" }`
- `POST /api/analyze/profile` - multipart form with `image` file (jpg/png/webp)
- `POST /api/chat` - LangGraph agent entrypoint. Body: `{ "userId": "...", "threadId": "...", "message": "text", "ui_events": [] }`
  - Keeps per-thread conversation state on the server (clarifications, disambiguations).
  - Legacy `/api/analyze/text` now wraps this agent and returns a simplified payload for older clients.
  - Config lives in `backend/config.py` with defaults for Qdrant/Tavily; override via environment variables as needed.

## Frontend setup (Next.js)
```bash
cd frontend
npm install
# or `pnpm install` / `yarn`

# Environment
# place this in frontend/.env.local so Next.js exposes it to the client
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000

npm run dev
```

Key UI pieces:
- Chat-style hero surface inspired by ChatGPT/Perplexity
- Search bar that posts to `/api/analyze/text`
- Profile upload CTA that posts the portrait to `/api/analyze/profile`
- Structured results displayed for both text search and profile dossier

## Notes
- Background uses light imagery + gradients to keep the fashion-first tone.
- CORS is open for local dev; tighten for production.
- The backend raises clear HTTP errors if the Responses API payload is malformed or missing JSON.

## Next steps
- Wire in your product catalog to convert keywords/palettes into shoppable results.
- Add auth + persistence for saved dossiers and histories.
