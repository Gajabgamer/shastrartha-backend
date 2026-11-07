import asyncio
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api.routes import api, ws
from .config import get_settings
from .knowledge import get_knowledge
from .engine.pvp import lobby
from .rag import dense, verify


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        get_knowledge()  # fail fast if data/ is incomplete
    except Exception as e:
        print(f"Warning: Knowledge loading failed: {e}")
    
    try:
        verify.warm_up()  # NLI loads in the background; lexical check covers the gap
    except Exception as e:
        print(f"Warning: NLI warm-up failed: {e}")
    
    # Open Chroma and load the embedding model before the first debate asks for it.
    try:
        threading.Thread(target=lambda: dense.search("chanakya", "warm up", k=1), name="dense-warmup", daemon=True).start()
    except Exception as e:
        print(f"Warning: Dense retrieval warm-up failed: {e}")
    
    try:
        ticker = asyncio.create_task(lobby.ticker())  # online matchmaking: widen Elo windows, pair players
    except Exception as e:
        print(f"Warning: Lobby ticker failed: {e}")
        ticker = None
    
    yield
    
    if ticker:
        ticker.cancel()


app = FastAPI(title="Shastrartha API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[get_settings().frontend_url],
    # any local dev port, plus private LAN addresses so audience phones on the same Wi-Fi can vote (Live Sabha)
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1|192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],  # includes x-guest-id, x-player-id, x-player-secret for online play
)


@app.exception_handler(RequestValidationError)
async def validation_error(_, exc: RequestValidationError):
    msg = "; ".join(f"{'.'.join(str(p) for p in e['loc'][1:])}: {e['msg']}" for e in exc.errors())
    return JSONResponse({"error": {"code": "VALIDATION_ERROR", "message": msg}}, status_code=422)


@app.get("/health")
def health():
    return {"status": "ok"}


from .api import glossary as glossary_api, report as report_api, study as study_api  # noqa: E402

app.include_router(api)
app.include_router(ws)
app.include_router(study_api.router)
app.include_router(report_api.router)
app.include_router(glossary_api.router)
