import time
import threading
from enum import Enum
from functools import wraps

import httpx

from src.utils.logger import get_logger

logger = get_logger("Resilience")

RETRYABLE_EXCEPTIONS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.TimeoutException,
    ConnectionError,
)

MAX_RETRY_ATTEMPTS = 5
BASE_BACKOFF_SECONDS = 1.0 # 1s, 2s, 4s, 8s, 16s

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitOpenError(Exception):
    """Raise when circuit breaker is OPEN and a call is rejected
        without even attemping to reach AI."""
    pass

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, cooldown_seconds: float = 30.0):
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._opened_at = None
        self._lock = threading.Lock()

    def _maybe_transition_to_half_open(self):
        if self._state == CircuitState.OPEN and self._opened_at is not None:
            elapsed = time.monotonic() - self._opened_at
            if elapsed >= self.cooldown_seconds:
                logger.info(
                    f"Circuit breaker cooldown elapsed ({elapsed:.1f}s)."
                    "Moving to HALF_OPEN -- next call will test the connection."
                )
                self._state = CircuitState.HALF_OPEN

    def before_call(self):
        with self._lock:
            self._maybe_transition_to_half_open()
            if self._state == CircuitState.OPEN:
                remaining_cooldown = self.cooldown_seconds - (time.monotonic() - self._opened_at) if self._opened_at is not None else self.cooldown_seconds
                raise CircuitOpenError(
                    f"Circuit breaker is OPEN after "
                    f"{self._consecutive_failures} consecutive failures. "
                    f"Failing fast without contacting AI server. Will retry "
                    f"the connection in "
                    f"{remaining_cooldown:.0f}s."
                )

    def record_success(self):
        with self._lock:
            if self._state != CircuitState.CLOSED:
                logger.info("AI call succeeded -- closing circuit breaker.")
            self._state = CircuitState.CLOSED
            self._consecutive_failures = 0
            self._opened_at = None

    def record_failure(self):
        with self._lock:
            self._consecutive_failures += 1
            if self._state == CircuitState.HALF_OPEN:
                logger.warning("HALF_OPEN test call failed. Reopen circuit breaker.")
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
            elif self._consecutive_failures >= self.failure_threshold:
                logger.warning(
                    f"{self._consecutive_failures} consecutive failures. "
                    f"Opening circuit breaker for {self.cooldown_seconds}s."
                )
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._maybe_transition_to_half_open()
            return self._state

circuit_breaker = CircuitBreaker()

def retry_with_backoff(func, *args, **kwargs):
    for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
        try:
            return func(*args, **kwargs)
        except RETRYABLE_EXCEPTIONS as e:
            last_exception = e
            if attempt == MAX_RETRY_ATTEMPTS:
                logger.error(
                    f"AI call failed after {MAX_RETRY_ATTEMPTS} attempts: {e}"
                )
                raise
            backoff = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                f"AI call failed (attempt {attempt}/{MAX_RETRY_ATTEMPTS}): {e}. "
                f"Retrying in {backoff:.0f}s."
            )
            time.sleep(backoff)

def with_resilience(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        circuit_breaker.before_call()
        try:
            result = retry_with_backoff(func, *args, **kwargs)
        except RETRYABLE_EXCEPTIONS:
            circuit_breaker.record_failure()
            raise
        else:
            circuit_breaker.record_success()
            return result
    return wrapper