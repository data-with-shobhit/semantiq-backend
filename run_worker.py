"""Celery worker entrypoint for Cloud Run.

Cloud Run requires an HTTP server listening on PORT before the health check
passes. HTTPServer binds the socket synchronously in __init__, so the port is
open before Celery starts — no race condition.
"""
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args):
        pass


port = int(os.environ.get("PORT", 8080))
server = HTTPServer(("0.0.0.0", port), HealthHandler)  # binds socket immediately
threading.Thread(target=server.serve_forever, daemon=True).start()

subprocess.run([
    "celery", "-A", "ingestion.tasks", "worker",
    "--loglevel=INFO",
    "--concurrency=2",
    "--without-heartbeat",
])
