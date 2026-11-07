# Shastrartha Backend

AI gurukul debate backend for [Shastrartha](https://shastrartha.vercel.app).

FastAPI server with RAG retrieval, fact-checking, debate orchestration, and Live Sabha voting.

## Features

- **Debate Engine:** LangGraph state machine with Purvapaksha → Khandana → Uttarapaksha stages
- **RAG Pipeline:** BM25 + optional ChromaDB dense retrieval with RRF fusion
- **Fact Verification:** NLI model (DeBERTa) or LLM-as-judge fallback
- **Scoring:** Pancavayava structure analysis, fallacy detection, pramana fit
- **Learner Model:** Elo per pramana + BKT per concept
- **Territory System:** Region capture by debate victory
- **Live Sabha:** QR-code voting for audience participation
- **Voice:** Fish Audio TTS with streaming + Gemini STT

## Deployment

### Deploy to Render

1. Fork this repo
2. Create new Web Service on [Render](https://render.com)
3. Connect your fork
4. Render auto-detects `render.yaml`
5. Add these environment variables in the Render dashboard:

```bash
# Required
GEMINI_API_KEYS=your-gemini-key-here
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
FRONTEND_URL=https://shastrartha.vercel.app

# Optional but recommended
NLI_ENABLED=false  # Set to false on free tier (512MB RAM limit)
DENSE_RETRIEVAL=false  # Set to false if no ChromaDB
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.5-flash-latest-exp
FISH_API_KEY=your-fish-key
FISH_TTS_MODEL=s2.1-pro-free
TTS_PROVIDER=fish
TTS_VOICE_CHANAKYA=b1c3b5492bd042999077081209b7eff6
TTS_VOICE_NAPOLEON=4022ee05e7b34cc39250e917dc03c172
TTS_VOICE_SUN_TZU=fe2e4e24a3a544dc95e8e5eb7dba7c82
TTS_VOICE_MARCUS_AURELIUS=52c76ed0d88546b69aaa0f1a88a28ea5
```

### Update Frontend

Once deployed, update the frontend environment:

```bash
cd frontend
npx vercel env add NEXT_PUBLIC_API_URL production
# Enter your Render URL: https://shastrartha-backend.onrender.com
npx vercel --prod
```

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables in .env
cp .env.example .env

# Run the server
uvicorn app.main:app --reload --port 8000
```

## API Endpoints

- `GET /api/health` - Health check with component status
- `GET /api/personas` - List available personas
- `POST /api/debates` - Start new debate
- `POST /api/debates/{id}/turns` - Submit argument
- `POST /api/debates/{id}/finish` - Finalize debate
- `GET /api/debates/{id}` - Get debate state
- `WS /ws/debates/{id}` - Real-time debate events
- `POST /api/sabha/{code}/vote` - Cast Sabha vote
- `GET /api/today` - Today's news briefing
- Many more... see `app/api/routes.py`

## Architecture

```
FastAPI
  ├── LangGraph orchestrator
  │     ├── Persona agent (cited replies)
  │     ├── Referee agent (scoring)
  │     ├── Fallacy detector
  │     └── NLI verifier
  ├── RAG (BM25 + ChromaDB)
  ├── Learner engine (Elo + BKT)
  ├── Territory system
  └── WebSocket events
```

## Free Tier Limitations

**Render Free (512 MB RAM):**
- ❌ NLI model (1.5 GB) - falls back to LLM verification
- ❌ ChromaDB not included - BM25-only retrieval
- ⚠️ Sleeps after 15 min inactivity (30-90s wake time)
- ✅ Voice cache regenerates on demand
- ✅ Debates persist in Supabase

## Tech Stack

- **Framework:** FastAPI + Uvicorn
- **Orchestration:** LangGraph
- **RAG:** BM25 (rank_bm25) + ChromaDB + sentence-transformers
- **NLI:** DeBERTa-v3-base-mnli-fever-anli (transformers + torch)
- **LLM:** Gemini 2.5 Flash (fallback: Flash 8B lite)
- **Database:** Supabase (Postgres + pgvector)
- **Voice:** Fish Audio API

## License

MIT

## Links

- **Frontend:** https://shastrartha.vercel.app
- **Docs:** See `docs/` directory in main repo
- **Handbook:** 29 personas × 17 regions × 11 topics
