"""Tests for metrics and timing utilities."""

import sys
import time

import pytest

sys.path.insert(0, str(pytest.importorskip("pathlib").Path(__file__).parent.parent))

from metrics import Timer, timed


class TestTimedDecorator:
    """Test suite for @timed decorator."""

    def test_preserves_function_name(self) -> None:
        """Test that decorator preserves function name."""

        @timed
        def my_function() -> None:
            pass

        assert my_function.__name__ == "my_function"

    def test_preserves_return_value(self) -> None:
        """Test that decorator preserves return value."""

        @timed
        def returns_value() -> int:
            return 42

        assert returns_value() == 42

    def test_preserves_return_value_with_args(self) -> None:
        """Test that decorator preserves return value with arguments."""

        @timed
        def add(a: int, b: int) -> int:
            return a + b

        assert add(2, 3) == 5

    def test_handles_exceptions(self) -> None:
        """Test that decorator handles exceptions properly."""

        @timed
        def raises_error() -> None:
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            raises_error()

    def test_handles_kwargs(self) -> None:
        """Test that decorator handles keyword arguments."""

        @timed
        def greet(name: str, greeting: str = "Hello") -> str:
            return f"{greeting}, {name}!"

        assert greet("World") == "Hello, World!"
        assert greet("World", greeting="Hi") == "Hi, World!"


class TestTimer:
    """Test suite for Timer context manager."""

    def test_timer_records_elapsed_time(self) -> None:
        """Test that timer records elapsed time."""
        with Timer("test_operation") as timer:
            time.sleep(0.01)  # Sleep for 10ms

        # Should be at least 10ms
        assert timer.elapsed >= 0.01
        # Should be less than 100ms (allowing for some overhead)
        assert timer.elapsed < 0.1

    def test_timer_name_is_set(self) -> None:
        """Test that timer name is set correctly."""
        with Timer("my_timer") as timer:
            pass

        assert timer.name == "my_timer"

    def test_timer_start_time_is_set(self) -> None:
        """Test that start time is set during context entry."""
        timer = Timer("test")
        assert timer.start_time == 0.0

        with timer:
            assert timer.start_time > 0.0

    def test_timer_elapsed_zero_before_context(self) -> None:
        """Test that elapsed is zero before context."""
        timer = Timer("test")
        assert timer.elapsed == 0.0

    def test_timer_elapsed_set_after_context(self) -> None:
        """Test that elapsed is set after context exit."""
        timer = Timer("test")

        with timer:
            time.sleep(0.005)

        assert timer.elapsed > 0.0

    def test_nested_timers(self) -> None:
        """Test that nested timers work correctly."""
        with Timer("outer") as outer:
            time.sleep(0.01)
            with Timer("inner") as inner:
                time.sleep(0.01)

        # Inner should be around 10ms
        assert inner.elapsed >= 0.01
        # Outer should be at least 20ms (inner + outer)
        assert outer.elapsed >= 0.02
        # Outer should be greater than inner
        assert outer.elapsed > inner.elapsed
