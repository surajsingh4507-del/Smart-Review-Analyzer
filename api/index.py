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
    headers = dict(scope.get("headers", []))

    # 1. Check x-invoke-path (Vercel serverless / AWS Lambda path)
    invoke_path = headers.get(b"x-invoke-path", b"").decode("latin1", errors="ignore").strip()
    if invoke_path and not invoke_path.startswith("/api/index.py"):
        real_path = invoke_path

    # 2. Check x-forwarded-uri (Vercel reverse proxy original URI)
    if not real_path:
        fwd_uri = headers.get(b"x-forwarded-uri", b"").decode("latin1", errors="ignore").strip()
        if fwd_uri and not fwd_uri.startswith("/api/index.py"):
            real_path = fwd_uri

    # 3. Check query string for __route_path__
    qs_bytes = scope.get("query_string", b"")
    qs = qs_bytes.decode("latin1", errors="ignore")
    if not real_path and "__route_path__" in qs:
        params = urllib.parse.parse_qs(qs, keep_blank_values=True)
        if "__route_path__" in params:
            route_vals = params.pop("__route_path__")
            if route_vals and route_vals[0]:
                real_path = route_vals[0]
                if "?" in real_path:
                    real_path, extra_qs = real_path.split("?", 1)
                    extra_params = urllib.parse.parse_qs(extra_qs, keep_blank_values=True)
                    for k, v in extra_params.items():
                        params.setdefault(k, []).extend(v)
                new_qs = urllib.parse.urlencode(params, doseq=True)
                scope["query_string"] = new_qs.encode("latin1")

    # 4. Check x-matched-path header (may contain __route_path__ or real path)
    if not real_path:
        matched = headers.get(b"x-matched-path", b"").decode("latin1", errors="ignore").strip()
        if "__route_path__" in matched:
            if "?" in matched:
                _, mqs = matched.split("?", 1)
                mparams = urllib.parse.parse_qs(mqs, keep_blank_values=True)
                if "__route_path__" in mparams and mparams["__route_path__"]:
                    real_path = mparams["__route_path__"][0]
        elif matched and not matched.startswith("/api/index.py"):
            real_path = matched

    # 5. Check x-now-route-matches (Vercel route match capture group $1)
    if not real_path:
        route_matches = headers.get(b"x-now-route-matches", b"").decode("latin1", errors="ignore").strip()
        if route_matches:
            rm_params = urllib.parse.parse_qs(route_matches, keep_blank_values=True)
            if "1" in rm_params and rm_params["1"]:
                real_path = "/" + rm_params["1"][0].lstrip("/")

    # 6. Normalize resolved path
    if real_path:
        if "?" in real_path:
            real_path, extra_qs = real_path.split("?", 1)
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
