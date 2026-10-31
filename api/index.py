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

async def app(scope, receive, send):
    if scope.get("type") == "http":
        qs_bytes = scope.get("query_string", b"")
        qs = qs_bytes.decode("latin1", errors="ignore")
        if "__route_path__" in qs:
            params = urllib.parse.parse_qs(qs, keep_blank_values=True)
            if "__route_path__" in params:
                real_path = params.pop("__route_path__")[0]
                if not real_path.startswith("/"):
                    real_path = "/" + real_path
                scope["path"] = real_path
                scope["raw_path"] = real_path.encode("latin1")
                new_qs = urllib.parse.urlencode(params, doseq=True)
                scope["query_string"] = new_qs.encode("latin1")
        elif scope.get("path") in ["/api/index.py", "/api/index", "/api", "/api/"]:
            scope["path"] = "/"
            scope["raw_path"] = b"/"

    await fastapi_app(scope, receive, send)
