import sys
import os
from pathlib import Path

# Add project root directory and api directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
api_dir = Path(__file__).resolve().parent

for p in [str(root_dir), str(api_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from main import app as fastapi_app

async def app(scope, receive, send):
    if scope.get("type") == "http":
        headers = dict(scope.get("headers", []))
        matched = (
            headers.get(b"x-matched-path")
            or headers.get(b"x-forwarded-uri")
            or headers.get(b"x-now-route-matches")
        )
        if matched:
            orig = matched.decode("latin1", errors="ignore")
            if "?" in orig:
                orig, qs = orig.split("?", 1)
                scope["query_string"] = qs.encode("latin1")
            # Only use if it does not point directly to index.py
            if orig and not orig.startswith("/api/index.py"):
                scope["path"] = orig
                scope["raw_path"] = orig.encode("latin1")
            elif orig in ["", "/", "/api/index.py"]:
                scope["path"] = "/"
                scope["raw_path"] = b"/"
        elif scope.get("path") in ["/api/index.py", "/api/index", "/api", "/api/"]:
            scope["path"] = "/"
            scope["raw_path"] = b"/"
    await fastapi_app(scope, receive, send)
