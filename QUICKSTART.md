# Render Deployment - Quick Start

**5-minute deployment for Shastrartha backend**

## Prerequisites

- GitHub account
- Render account: https://render.com (sign up free)
- Gemini API key: https://aistudio.google.com/apikey
- Supabase project URL + service role key

## Step 1: Push to GitHub (2 min)

```powershell
cd d:\sm\shastrartha-render

git init
git add .
git commit -m "Deploy to Render"

# Create new repo on GitHub: https://github.com/new
# Name it: shastrartha-backend
# Then:

git remote add origin https://github.com/YOUR-USERNAME/shastrartha-backend.git
git branch -M main
git push -u origin main
```

## Step 2: Deploy on Render (3 min)

1. Go to https://render.com/dashboard
2. **New +** → **Web Service**
3. Connect GitHub → Select `shastrartha-backend`
4. Render auto-detects `render.yaml` ✅
5. Click **Environment** tab, add these:

```bash
# Essential (copy-paste these)
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.5-flash-latest-exp
GEMINI_API_KEYS=YOUR-GEMINI-KEY-HERE
SUPABASE_URL=YOUR-SUPABASE-URL
SUPABASE_SERVICE_ROLE_KEY=YOUR-SERVICE-KEY
FRONTEND_URL=https://shastrartha.vercel.app
NLI_ENABLED=false
DENSE_RETRIEVAL=false

# Optional (voice)
TTS_PROVIDER=fish
FISH_API_KEY=YOUR-FISH-KEY
FISH_TTS_MODEL=s2.1-pro-free
TTS_VOICE_CHANAKYA=b1c3b5492bd042999077081209b7eff6
TTS_VOICE_NAPOLEON=4022ee05e7b34cc39250e917dc03c172
TTS_VOICE_SUN_TZU=fe2e4e24a3a544dc95e8e5eb7dba7c82
TTS_VOICE_MARCUS_AURELIUS=52c76ed0d88546b69aaa0f1a88a28ea5
```

6. **Create Web Service**
7. Wait 3-5 minutes for build
8. Copy your URL: `https://shastrartha-backend.onrender.com`

## Step 3: Update Frontend (1 min)

```powershell
cd d:\sm\Sashtrartha\shastrartha\shastrartha\frontend

npx vercel env rm NEXT_PUBLIC_API_URL production
npx vercel env add NEXT_PUBLIC_API_URL production
# Paste: https://shastrartha-backend.onrender.com

npx vercel --prod
```

## Step 4: Test

Open https://shastrartha.vercel.app
- Play → Pick guru → Start debate
- Should see citations, scoring, voice (if enabled)

**First request after 15 min sleep takes 30-90s to wake up!**

## Troubleshooting

### Build fails
Check Render logs. Most common: missing env variable.

### Frontend can't connect
1. Verify `FRONTEND_URL` in Render matches Vercel domain exactly
2. Wait 2 minutes for cold start
3. Check browser console for CORS errors

### Voice doesn't work
Verify `TTS_PROVIDER=fish` and `FISH_API_KEY` are set.

### Want local testing?
```powershell
cd d:\sm\shastrartha-render
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your keys
uvicorn app.main:app --reload
```

## What's Different from Local?

**Free Tier (512 MB RAM):**
- ✅ Core debates work perfectly
- ✅ BM25 retrieval (curated 629 chunks)
- ✅ LLM verification (replaces NLI)
- ✅ Voice generation
- ✅ Sabha voting
- ❌ No DeBERTa NLI model (uses LLM instead)
- ❌ No ChromaDB library (BM25 only)
- ⚠️ Sleeps after 15 min inactivity

## Need Help?

- See `DEPLOY.md` for detailed guide
- Render docs: https://render.com/docs
- Alternative: Railway.app (also free)

---

**Total time: ~8 minutes** ⚡
