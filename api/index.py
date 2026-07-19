import sys
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.main import app  # noqa: E402,F401

FRONTEND_DIST = ROOT / "frontend" / "dist"
FRONTEND_INDEX = FRONTEND_DIST / "index.html"
FRONTEND_ASSETS = FRONTEND_DIST / "assets"
FRONTEND_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; "
        "connect-src 'self' https://vercel.live; "
        "img-src 'self' data: blob:; "
        "style-src 'self' 'unsafe-inline'; "
        "script-src 'self' https://vercel.live; "
        "font-src 'self'; "
        "frame-ancestors 'none'; "
        "base-uri 'self'; "
        "form-action 'self'"
    )
}

if FRONTEND_ASSETS.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_ASSETS), name="assets")


@app.get("/", include_in_schema=False)
@app.get("/{path:path}", include_in_schema=False)
async def serve_frontend(path: str = "") -> FileResponse:
    """Serve the Vite SPA shell for Vercel frontend routes."""
    if path.startswith("api/"):
        raise HTTPException(status_code=404, detail="Not Found")

    requested_file = FRONTEND_DIST / path
    if path and requested_file.is_file():
        return FileResponse(requested_file, headers=FRONTEND_HEADERS)

    if FRONTEND_INDEX.is_file():
        return FileResponse(FRONTEND_INDEX, headers=FRONTEND_HEADERS)

    raise HTTPException(status_code=404, detail="Frontend build was not found.")
