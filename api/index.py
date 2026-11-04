import sys
import os
import urllib.parse
from pathlib import Path

# Add project root directory and api directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
api_dir = Path(__file__).resolve().parent

for p in [str(root_dir), str(api_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from main import app as fastapi_app


def unpack_asgi_scope(scope: dict) -> dict:
    """
    ASGI scope unpacker for Vercel Serverless Function deployment.
    Unpacks the rewritten request path and query string so FastAPI receives
    the actual intended route (e.g. /api/analyze, /docs, /robots.txt) rather
    than /api/index.py, avoiding HTTP 405 Method Not Allowed errors.
    """
    if scope.get("type") != "http":
        return scope

    real_path = None
    qs_bytes = scope.get("query_string", b"")
    qs = qs_bytes.decode("latin1", errors="ignore")

    # 1. First priority: Check query string for __route_path__ injected by vercel.json
    if "__route_path__" in qs:
        params = urllib.parse.parse_qs(qs, keep_blank_values=True)
        if "__route_path__" in params:
            route_vals = params.pop("__route_path__")
            if route_vals and route_vals[0]:
                real_path = route_vals[0]
                # In case the route path itself contained an embedded query string
                if "?" in real_path:
                    real_path, extra_qs = real_path.split("?", 1)
                    extra_params = urllib.parse.parse_qs(extra_qs, keep_blank_values=True)
                    for k, v in extra_params.items():
                        params.setdefault(k, []).extend(v)
                # Re-encode query string without __route_path__
                new_qs = urllib.parse.urlencode(params, doseq=True)
                scope["query_string"] = new_qs.encode("latin1")

    # 2. Second priority: Check Vercel routing headers
    if not real_path:
        headers = dict(scope.get("headers", []))
        matched = (
            headers.get(b"x-matched-path")
            or headers.get(b"x-forwarded-uri")
            or headers.get(b"x-real-path")
        )
        if matched:
            candidate = matched.decode("latin1", errors="ignore")
            if "?" in candidate:
                candidate, _ = candidate.split("?", 1)
            if candidate and not candidate.startswith("/api/index.py"):
                real_path = candidate

    # 3. Normalize and apply resolved path
    if real_path:
        real_path = "/" + real_path.lstrip("/")
        if real_path in ["/api/index.py", "/api/index", "/api", "/api/"]:
            real_path = "/"
        scope["path"] = real_path
        scope["raw_path"] = real_path.encode("latin1")
    elif scope.get("path") in ["/api/index.py", "/api/index", "/api", "/api/"]:
        scope["path"] = "/"
        scope["raw_path"] = b"/"

    return scope


async def app(scope, receive, send):
    """
    ASGI entrypoint for Vercel Serverless Function deployment.
    """
    if scope.get("type") == "http":
        unpack_asgi_scope(scope)
    await fastapi_app(scope, receive, send)
