"""
AIModelRouter — Meta Model Server

Entry point for the FastAPI application.
Initializes shared instances (registry, selector, dispatcher) on startup
and makes them available to routes via app.state.
"""

import logging
import os
import secrets
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from .config import settings
from .conversation_store import ConversationStore
from .logging_util import ProjectLogger
from .model_dispatcher import ModelDispatcher
from .model_selector import ModelSelector
from .provider_registry import ProviderRegistry
from .usage_tracker import UsageTracker
from .router import api_router
from .admin import admin_router

# Configure logging early
ProjectLogger.configure(
    project_name="RelayFreeLLM",
    log_dir="logs",
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
)
logger = ProjectLogger.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle for shared state."""
    logger.info("=== RelayFreeLLM starting up ===")

    # Say plainly which surface this instance is serving. A gateway that is open
    # when you believed it keyed is the failure this fork exists to prevent, and
    # it is invisible unless something says so at boot.
    if RELAY_ALLOW_UI:
        logger.warning(
            "RELAY_ALLOW_UI=1 — every route is served and /v1 needs no key. "
            "Correct for localhost; NEVER for a public deployment."
        )
    elif not RELAY_CLIENT_KEY:
        logger.warning(
            "RELAY_CLIENT_KEY is unset — the UI and admin routes are 404, but "
            "/v1/* answers ANY caller. Set it here and set the same value as "
            "RELAYFREE_API_KEY on the consumer."
        )
    else:
        logger.info("surface: /health open, /v1/* keyed, everything else 404")

    # 1. Auto-discover provider clients (This asserts Python code & API keys are valid)
    registry = ProviderRegistry()
    registry.auto_discover()

    # 2. Initialize model selector (This asserts limits exist in JSON)
    selector = ModelSelector()

    # --- SYNCHRONIZE AND VALIDATE ---
    registered_providers = set(registry.list_providers())
    json_providers = set(selector.providers.keys())

    active_providers = registered_providers.intersection(json_providers)

    # Scour out unsupported providers
    for p in list(registered_providers):
        if p not in active_providers:
            logger.warning(f"Provider '{p}' has credentials/code but lacks JSON limits! Pruning.")
            registry.unregister(p)

    for p in list(json_providers):
        if p not in active_providers:
            logger.warning(f"Provider '{p}' has JSON limits but lacks credentials/code! Pruning.")
            selector.remove_provider(p)

    # ABORT IF EMPTY
    if len(active_providers) == 0:
        logger.critical("No valid providers registered (check .env and JSON limits)! Aborting server startup.")
        sys.exit(1)

    # 3. Initialize conversation store (persisted conversations)
    conversation_store = ConversationStore()

    # 4. Initialize usage tracker (persisted stats)
    usage_tracker = UsageTracker()

    # 4. Create the dispatcher (the meta model core)
    dispatcher = ModelDispatcher(
        registry=registry, 
        selector=selector, 
        usage_tracker=usage_tracker
    )

    # Inject into app.state so routes can access them
    app.state.registry = registry
    app.state.selector = selector
    app.state.dispatcher = dispatcher
    app.state.conversation_store = conversation_store
    app.state.usage_tracker = usage_tracker

    logger.info(
        f"Meta model '{settings.META_MODEL_NAME}' ready with providers: "
        f"{registry.list_providers()}"
    )

    yield  # app is running

    logger.info("=== RelayFreeLLM shutting down ===")


app = FastAPI(
    title="RelayFreeLLM — Meta Model",
    description="A unified LLM endpoint that transparently routes across multiple AI providers.",
    version="2.0.0",
    lifespan=lifespan,
)

# ─── Fork patch: restrict the public surface ─────────────────────────────────
#
# Upstream RelayFreeLLM has no client authentication, and on localhost:8000 that
# is the right call. This fork exists because ours runs on a PUBLIC URL, where
# the same defaults published the bundled chat UI, the admin API — including
# `PUT /admin/api/limits`, which can RAISE the provider limits and make the
# gateway over-call the upstream accounts — and the conversation store to anyone
# who found the subdomain.
#
# An allowlist rather than a bearer check bolted onto every route, because the
# one consumer of this deployment uses exactly two paths:
#
#   GET  /health                 the keep-warm ping, the host's health check, and
#                                how a human wakes a spun-down free instance
#   POST /v1/chat/completions    the only thing creai actually calls
#
# Everything else answers 404 rather than 401: nothing needs it, so there is no
# reason to confirm to a scanner that it exists.
#
# Two switches, and the DEFAULT is locked, deliberately — an unset variable
# should fail safe:
#
#   RELAY_ALLOW_UI=1    restore upstream behaviour wholesale (local development)
#   RELAY_CLIENT_KEY    require `Authorization: Bearer <key>` on /v1/*
#
# The route lockdown does NOT depend on RELAY_CLIENT_KEY. That is what lets this
# be deployed before the key exists on both sides: the large exposure (the UI and
# the admin API) closes immediately, while /v1 keeps serving so generation does
# not break in the gap. Startup logs loudly while that gap is open.

RELAY_ALLOW_UI = os.getenv("RELAY_ALLOW_UI", "").strip() == "1"
RELAY_CLIENT_KEY = os.getenv("RELAY_CLIENT_KEY", "").strip()


@app.middleware("http")
async def restrict_public_surface(request: Request, call_next):
    if RELAY_ALLOW_UI:
        return await call_next(request)

    path = request.url.path

    # The wake-up page. Upstream has no "/" route at all, so this URL always
    # answered FastAPI's bare {"detail":"Not Found"} — and this deployment sleeps
    # after 15 idle minutes, so opening it in a browser to wake it is a normal
    # thing to do. A 404 does wake the instance, but it reads like a broken
    # service, which is a bad answer to "is it up?".
    #
    # Deliberately says nothing about what else is here.
    if path == "/":
        return JSONResponse({"status": "awake"})

    if path == "/health":
        return await call_next(request)

    if path.startswith("/v1/"):
        if RELAY_CLIENT_KEY:
            # compare_digest, not ==, so a wrong key cannot be recovered a
            # character at a time from response timing.
            presented = request.headers.get("authorization", "")
            if not secrets.compare_digest(presented, f"Bearer {RELAY_CLIENT_KEY}"):
                return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)

    return JSONResponse({"detail": "Not Found"}, status_code=404)


app.add_middleware(
    CORSMiddleware,
    # No browser origin needs this deployment: the only consumer is a server.
    # Upstream's ["*"] with allow_credentials=True is the browser half of the
    # same hole (and is rejected by browsers anyway). RELAY_ALLOW_UI restores it
    # for local use, where the bundled UI is the point.
    allow_origins=["*"] if RELAY_ALLOW_UI else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(api_router)
app.include_router(admin_router)


if __name__ == "__main__":
    uvicorn.run(
        app,
        port=settings.PORT,
        host=settings.HOST,
    )
