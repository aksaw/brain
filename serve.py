"""Serve this folder on http://localhost:8000 and tell the browser never to cache.

Browsers otherwise keep stale copies of JavaScript modules between edits; the audio
worklet is especially prone to it, which leaves the page running old sound code.

    python web/serve.py [port]
"""

import functools
import http.server
import pathlib
import sys


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
handler = functools.partial(NoCacheHandler, directory=str(pathlib.Path(__file__).parent))
print(f"Serving on http://localhost:{port}")
http.server.ThreadingHTTPServer(("localhost", port), handler).serve_forever()
