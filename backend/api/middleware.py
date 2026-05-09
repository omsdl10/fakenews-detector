"""
Custom ASGI middleware:
  - LoggingMiddleware: structured request/response logging with latency
  - RateLimitMiddleware: simple in-memory sliding-window rate limiter per IP
    (use Redis-backed rate limiting in multi-process deployments)
"""

import time
from collections import defaultdict, deque
from typing import Callable, Deque, Dict

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = structlog.get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Log every HTTP request with method, path, status, and latency.
    Skips /health to avoid noise in production logs.
    """

    SKIP_PATHS = {"/health", "/favicon.ico"}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Bind request context for all downstream log statements
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request.headers.get("X-Request-ID", "-"),
            method=request.method,
            path=request.url.path,
        )

        start = time.perf_counter()
        response = await call_next(request)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        if not any(request.url.path.endswith(p) for p in self.SKIP_PATHS):
            logger.info(
                "HTTP request",
                status=response.status_code,
                latency_ms=latency_ms,
            )

        response.headers["X-Response-Time"] = f"{latency_ms}ms"
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding-window rate limiter per client IP.

    Default: 60 requests / minute.
    Returns HTTP 429 when exceeded.

    NOTE: This is a single-process in-memory implementation suitable for
    development and single-instance deployments. Use Redis + a token-bucket
    algorithm for multi-worker production deployments.
    """

    def __init__(self, app: ASGIApp, requests_per_minute: int = 60) -> None:
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.window_seconds = 60
        # {ip: deque of timestamps}
        self._windows: Dict[str, Deque[float]] = defaultdict(deque)

    def _get_client_ip(self, request: Request) -> str:
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        ip = self._get_client_ip(request)
        now = time.time()
        window = self._windows[ip]

        # Evict timestamps outside the sliding window
        while window and window[0] < now - self.window_seconds:
            window.popleft()

        if len(window) >= self.requests_per_minute:
            logger.warning("Rate limit exceeded", ip=ip, count=len(window))
            return Response(
                content='{"detail": "Rate limit exceeded. Please slow down."}',
                status_code=429,
                media_type="application/json",
                headers={
                    "Retry-After": str(self.window_seconds),
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                },
            )

        window.append(now)
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(
            self.requests_per_minute - len(window)
        )
        return response
