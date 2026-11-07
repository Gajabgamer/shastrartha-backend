# Render Deployment Guide

Step-by-step instructions to deploy Shastrartha backend to Render.com free tier.

## Prerequisites

1. GitHub account
2. Render account (free): https://render.com
3. Supabase project with the schema deployed
4. Gemini API key(s): https://aistudio.google.com/apikey
5. Optional: Fish Audio API key for voice: https://fish.audio

## Step 1: Push to GitHub

```powershell
# Navigate to this directory
cd d:\sm\shastrartha-render

# Initialize git
git init
git add .
git commit -m "Initial Render deployment"

# Create a new repository on GitHub named "shastrartha-backend"
# Then connect and push:
git remote add origin https://github.com/YOUR-USERNAME/shastrartha-backend.git
git branch -M main
git push -u origin main
```

## Step 2: Create Render Web Service

1. Go to https://render.com/dashboard
2. Click **New +** → **Web Service**
3. Connect your GitHub account (if not already)
4. Find and select your `shastrartha-backend` repository
5. Click **Connect**

Render will auto-detect the `render.yaml` configuration.

## Step 3: Configure Environment Variables

In the Render dashboard, go to **Environment** tab and add these variables:

### Required Variables

```bash
# LLM
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.5-flash-latest-exp
GEMINI_API_KEYS=your-gemini-key-here

# Database
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key

# CORS
FRONTEND_URL=https://shastrartha.vercel.app

# Memory optimization for free tier
NLI_ENABLED=false
DENSE_RETRIEVAL=false
```

### Optional Variables (Voice)

```bash
TTS_PROVIDER=fish
FISH_API_KEY=your-fish-api-key
FISH_TTS_MODEL=s2.1-pro-free
TTS_VOICE_CHANAKYA=b1c3b5492bd042999077081209b7eff6
TTS_VOICE_NAPOLEON=4022ee05e7b34cc39250e917dc03c172
TTS_VOICE_SUN_TZU=fe2e4e24a3a544dc95e8e5eb7dba7c82
TTS_VOICE_MARCUS_AURELIUS=52c76ed0d88546b69aaa0f1a88a28ea5
```

## Step 4: Deploy

1. Click **Create Web Service**
2. Render will build and deploy (takes 3-5 minutes first time)
3. Watch the logs for any errors
4. Once deployed, you'll get a URL like: `https://shastrartha-backend.onrender.com`

## Step 5: Verify Deployment

Test the health endpoint:

```bash
curl https://shastrartha-backend.onrender.com/api/health
```

Expected response:
```json
{
  "status": "ok",
  "nli": "disabled",
  "dense": false,
  "llm": true
}
```

## Step 6: Update Frontend

Update the Vercel frontend to use the new backend:

```powershell
cd d:\sm\Sashtrartha\shastrartha\shastrartha\frontend

# Remove old API URL
npx vercel env rm NEXT_PUBLIC_API_URL production

# Add new Render URL
npx vercel env add NEXT_PUBLIC_API_URL production
# When prompted, enter: https://shastrartha-backend.onrender.com

# Redeploy frontend
npx vercel --prod
```

## Step 7: Test End-to-End

1. Open https://shastrartha.vercel.app
2. Click **Play**
3. Select a guru and topic
4. Start a debate
5. Verify:
   - Citations appear
   - Scoring works
   - Voice plays (if enabled)
   - Sabha voting works (if testing with phones)

## Free Tier Limitations

**Render Free (512 MB RAM, 750 hours/month):**

✅ **Works:**
- Core debate engine
- BM25 retrieval (curated corpus only)
- LLM verification (replaces NLI)
- Scoring & territory
- Voice generation (cached)
- Sabha voting
- All API endpoints

⚠️ **Limitations:**
- **Sleep after 15 min:** First request takes 30-90s to wake up
- **No NLI model:** Uses LLM-as-judge instead (slower, uses Gemini quota)
- **No ChromaDB:** Falls back to BM25-only retrieval (still works well)
- **Ephemeral disk:** Voice cache and today's topics regenerate on restart
- **750 hours/month:** ~31 days if always running; perfect for demos

❌ **Doesn't fit:**
- DeBERTa NLI model (~1.5 GB)
- ChromaDB full library (~400 MB)

## Troubleshooting

### Build fails with "No module named 'transformers'"

✅ Expected! The lightweight `requirements.txt` intentionally excludes heavy dependencies.

### Service shows 503 or crashes

Check logs for memory issues. The free tier has 512 MB. If crashes persist:
1. Reduce concurrent users
2. Lower `LLM_MAX_TOKENS` to 500
3. Disable voice: `TTS_PROVIDER=` (leave empty)

### "NLI model unavailable" in logs

✅ Expected! `NLI_ENABLED=false` means it uses LLM verification instead.

### Frontend shows "Failed to fetch"

1. Check CORS: `FRONTEND_URL` must match your Vercel domain exactly
2. Check Render logs for errors
3. Verify environment variables are set
4. Wait 2 minutes for cold start wake-up

### Dense retrieval not working

✅ Expected! `DENSE_RETRIEVAL=false` on free tier. BM25 handles retrieval.

### Voice not generating

Check:
1. `FISH_API_KEY` is set correctly
2. `TTS_PROVIDER=fish` (not empty)
3. Voice IDs are correct
4. Fish quota hasn't run out

## Updating the Deployment

To push updates:

```powershell
cd d:\sm\shastrartha-render
git add .
git commit -m "Update backend"
git push
```

Render auto-deploys on push to `main` branch.

## Alternative: Railway

If Render free tier doesn't work, try Railway (also free with $5/month credits):

1. Sign up at https://railway.app
2. **New Project** → **Deploy from GitHub repo**
3. Select `shastrartha-backend`
4. Add the same environment variables in **Variables** tab
5. Railway auto-detects Python and deploys

URL format: `https://shastrartha-backend.up.railway.app`

## For Demo Day

**Best practice:** Run backend on your laptop during live demo for:
- Full NLI verification
- No cold-start delays  
- Fast voice generation
- ChromaDB dense retrieval

Deploy to Render/Railway for:
- Post-demo persistent link in your deck
- Judge testing after presentation
- Backup if laptop fails

## Support

- Render docs: https://render.com/docs
- Render community: https://community.render.com
- Railway docs: https://docs.railway.app

---

**Next:** Test the deployment, then update your pitch deck with the live URL!
