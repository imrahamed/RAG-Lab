"""Optional Langfuse observations with isolated telemetry failures and explicit content capture."""

import logging
from contextlib import contextmanager

from rag_lab.config import Settings

logger = logging.getLogger(__name__)


class Observation:
    def __init__(self, span=None, capture_content: bool = False):
        self.span = span
        self.capture_content = capture_content

    @property
    def trace_id(self) -> str | None:
        try:
            return self.span.trace_id if self.span is not None else None
        except Exception:
            return None

    def update(self, **attributes):
        if self.span is None:
            return
        if not self.capture_content:
            attributes.pop("input", None)
            attributes.pop("output", None)
        try:
            self.span.update(**attributes)
        except Exception as error:
            logger.warning("Langfuse update failed (%s)", type(error).__name__)


class Tracing:
    def __init__(self, settings: Settings | None = None):
        self.enabled = bool(settings and settings.langfuse_enabled)
        self.capture_content = bool(settings and settings.langfuse_capture_content)
        self.client = None
        self.error_type = None
        if self.enabled:
            try:
                self.client = self._create_client(settings)
            except Exception as error:
                self.error_type = type(error).__name__
                logger.warning("Langfuse initialization failed (%s)", self.error_type)

    @staticmethod
    def _create_client(settings: Settings):
        from langfuse import Langfuse
        from opentelemetry.sdk.trace import TracerProvider

        return Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key.get_secret_value(),
            base_url=str(settings.langfuse_base_url).rstrip("/"),
            environment=settings.langfuse_environment,
            timeout=settings.langfuse_timeout_seconds,
            tracing_enabled=True,
            # Keep this application's exporter separate from other OTEL instrumentation.
            tracer_provider=TracerProvider(),
        )

    @contextmanager
    def span(self, name: str, *, as_type: str = "span", **attributes):
        manager = None
        observation = Observation()
        if not self.capture_content:
            attributes.pop("input", None)
            attributes.pop("output", None)
        if self.client is not None:
            try:
                manager = self.client.start_as_current_observation(
                    name=name, as_type=as_type, **attributes
                )
                observation = Observation(manager.__enter__(), self.capture_content)
            except Exception as error:
                manager = None
                logger.warning("Langfuse observation failed (%s)", type(error).__name__)
        try:
            yield observation
        except BaseException as error:
            # Exception messages/stack traces can contain database URLs or credentials.
            observation.update(level="ERROR", status_message=type(error).__name__)
            raise
        finally:
            if manager is not None:
                try:
                    # Record sanitized error status above, without SDK exception auto-capture.
                    manager.__exit__(None, None, None)
                except Exception as error:
                    logger.warning("Langfuse observation close failed (%s)", type(error).__name__)

    def status(self, check_connection: bool = False) -> dict:
        if not self.enabled:
            return {"status": "disabled"}
        if self.client is None:
            return {"status": "error", "error_type": self.error_type}
        if check_connection:
            try:
                if not self.client.auth_check():
                    raise RuntimeError("Langfuse authentication failed")
            except Exception as error:
                return {"status": "error", "error_type": type(error).__name__}
        return {
            "status": "ok",
            "auth_checked": check_connection,
            "capture_content": self.capture_content,
        }

    def close(self):
        if self.client is not None:
            try:
                # CLI processes and the API lifespan both drain the buffered exporter here.
                self.client.shutdown()
            except Exception as error:
                logger.warning("Langfuse shutdown failed (%s)", type(error).__name__)
