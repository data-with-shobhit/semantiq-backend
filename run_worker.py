"""Celery worker entrypoint for Cloud Run.

Cloud Run requires an HTTP server on port 8080. This script starts a minimal
health-check server in a background thread, then runs the Celery worker in
the foreground.
"""
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


def _serve():
    HTTPServer(("0.0.0.0", 8080), HealthHandler).serve_forever()


threading.Thread(target=_serve, daemon=True).start()

subprocess.run([
    "celery", "-A", "ingestion.tasks", "worker",
    "--loglevel=INFO",
    "--concurrency=2",
    "--without-heartbeat",
])
