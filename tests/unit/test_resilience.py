"""
Unit tests for src/utils/resilience.py

Covers retry-with-backoff timing/exception scoping and the circuit
breaker's state machine (CLOSED -> OPEN -> HALF_OPEN -> CLOSED).
time.sleep is mocked throughout so these tests run in milliseconds,
not real seconds -- a real 1+2+4+8+16=31s test suite would be
unusable in a fast feedback loop.
"""
import time
from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.utils.resilience import (
    CircuitOpenError,
    CircuitState,
    MAX_RETRY_ATTEMPTS,
    CircuitBreaker,
    retry_with_backoff,
    with_resilience,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# retry_with_backoff
# ---------------------------------------------------------------------------


def test_retry_returns_result_on_first_success():
    func = MagicMock(return_value="ok")

    result = retry_with_backoff(func)

    assert result == "ok"
    assert func.call_count == 1


def test_retry_passes_through_args_and_kwargs():
    func = MagicMock(return_value="ok")

    retry_with_backoff(func, "positional", keyword="value")

    func.assert_called_once_with("positional", keyword="value")


@patch("src.utils.resilience.time.sleep")
def test_retry_succeeds_after_transient_failures(mock_sleep):
    func = MagicMock(side_effect=[httpx.ConnectError("refused"), httpx.ConnectError("refused"), "ok"])

    result = retry_with_backoff(func)

    assert result == "ok"
    assert func.call_count == 3


@patch("src.utils.resilience.time.sleep")
def test_retry_uses_exponential_backoff_delays(mock_sleep):
    func = MagicMock(side_effect=httpx.ConnectError("refused"))

    with pytest.raises(httpx.ConnectError):
        retry_with_backoff(func)

    delays = [call.args[0] for call in mock_sleep.call_args_list]
    assert delays == [1.0, 2.0, 4.0, 8.0]  # 4 sleeps between 5 attempts


@patch("src.utils.resilience.time.sleep")
def test_retry_gives_up_after_max_attempts(mock_sleep):
    func = MagicMock(side_effect=httpx.ConnectError("refused"))

    with pytest.raises(httpx.ConnectError):
        retry_with_backoff(func)

    assert func.call_count == MAX_RETRY_ATTEMPTS


def test_retry_does_not_retry_non_retryable_exceptions():
    func = MagicMock(side_effect=ValueError("bad input"))

    with pytest.raises(ValueError):
        retry_with_backoff(func)

    assert func.call_count == 1


@patch("src.utils.resilience.time.sleep")
def test_retry_retries_on_read_timeout(mock_sleep):
    func = MagicMock(side_effect=[httpx.ReadTimeout("timed out"), "ok"])

    result = retry_with_backoff(func)

    assert result == "ok"
    assert func.call_count == 2


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------


@pytest.fixture
def breaker():
    return CircuitBreaker(failure_threshold=3, cooldown_seconds=10.0)


def test_circuit_starts_closed(breaker):
    assert breaker.state == CircuitState.CLOSED


def test_before_call_does_not_raise_when_closed(breaker):
    breaker.before_call()  # should not raise


def test_circuit_stays_closed_below_failure_threshold(breaker):
    breaker.record_failure()
    breaker.record_failure()

    assert breaker.state == CircuitState.CLOSED


def test_circuit_opens_at_failure_threshold(breaker):
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_failure()

    assert breaker.state == CircuitState.OPEN


def test_before_call_raises_when_open(breaker):
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_failure()

    with pytest.raises(CircuitOpenError):
        breaker.before_call()


def test_success_resets_consecutive_failure_count(breaker):
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    breaker.record_failure()

    # Only 2 consecutive failures since the reset -- still below
    # threshold of 3, so circuit must still be closed.
    assert breaker.state == CircuitState.CLOSED


def test_circuit_transitions_to_half_open_after_cooldown():
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=0.05)
    breaker.record_failure()
    assert breaker.state == CircuitState.OPEN

    time.sleep(0.1)

    assert breaker.state == CircuitState.HALF_OPEN


def test_half_open_success_closes_circuit():
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=0.05)
    breaker.record_failure()
    time.sleep(0.1)
    assert breaker.state == CircuitState.HALF_OPEN

    breaker.record_success()

    assert breaker.state == CircuitState.CLOSED


def test_half_open_failure_reopens_circuit_and_restarts_cooldown():
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=0.05)
    breaker.record_failure()
    time.sleep(0.1)
    assert breaker.state == CircuitState.HALF_OPEN

    breaker.record_failure()

    assert breaker.state == CircuitState.OPEN
    # Cooldown must have restarted -- immediately after the HALF_OPEN
    # failure, before_call() should reject again rather than allow
    # another test call right away.
    with pytest.raises(CircuitOpenError):
        breaker.before_call()


def test_before_call_allows_test_call_during_half_open():
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=0.05)
    breaker.record_failure()
    time.sleep(0.1)

    breaker.before_call()  # should not raise -- HALF_OPEN allows one test call


# ---------------------------------------------------------------------------
# with_resilience (integration of retry + circuit breaker)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_shared_circuit_breaker():
    """
    with_resilience uses the module-level shared
    circuit_breaker singleton. Reset it before and after every
    test in this section so tests can't leak state into each other.
    """
    import src.utils.resilience as resilience_module

    resilience_module.circuit_breaker.record_success()  # force CLOSED
    yield
    resilience_module.circuit_breaker.record_success()


def test_with_resilience_returns_result_on_success():
    func = MagicMock(return_value="answer")
    wrapped = with_resilience(func)

    result = wrapped("query text")

    assert result == "answer"


@patch("src.utils.resilience.time.sleep")
def test_with_resilience_retries_then_succeeds(mock_sleep):
    func = MagicMock(side_effect=[httpx.ConnectError("refused"), "answer"])
    wrapped = with_resilience(func)

    result = wrapped("query text")

    assert result == "answer"
    assert func.call_count == 2


@patch("src.utils.resilience.time.sleep")
def test_with_resilience_opens_circuit_after_repeated_failures(mock_sleep):
    import src.utils.resilience as resilience_module

    func = MagicMock(side_effect=httpx.ConnectError("refused"))
    wrapped = with_resilience(func)

    # Each call exhausts MAX_RETRY_ATTEMPTS and then records one
    # circuit-breaker failure. Default threshold is 5.
    for _ in range(5):
        with pytest.raises(httpx.ConnectError):
            wrapped("query text")

    assert resilience_module.circuit_breaker.state == CircuitState.OPEN


@patch("src.utils.resilience.time.sleep")
def test_with_resilience_fails_fast_when_circuit_open(mock_sleep):
    import src.utils.resilience as resilience_module

    func = MagicMock(side_effect=httpx.ConnectError("refused"))
    wrapped = with_resilience(func)

    for _ in range(5):
        with pytest.raises(httpx.ConnectError):
            wrapped("query text")
    assert resilience_module.circuit_breaker.state == CircuitState.OPEN

    func.reset_mock()
    with pytest.raises(CircuitOpenError):
        wrapped("query text")

    # The underlying function must NOT have been called at all --
    # that's the whole point of failing fast.
    func.assert_not_called()


def test_with_resilience_success_resets_circuit_breaker():
    import src.utils.resilience as resilience_module

    resilience_module.circuit_breaker.record_failure()
    resilience_module.circuit_breaker.record_failure()

    func = MagicMock(return_value="answer")
    wrapped = with_resilience(func)
    wrapped("query text")

    assert resilience_module.circuit_breaker.state == CircuitState.CLOSED