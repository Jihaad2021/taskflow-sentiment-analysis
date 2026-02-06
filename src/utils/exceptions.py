"""Custom exceptions for TaskFlow."""


class TaskFlowError(Exception):
    """Base exception for TaskFlow errors."""

    pass


class ValidationError(TaskFlowError):
    """Raised when data validation fails."""

    pass


class AgentExecutionError(TaskFlowError):
    """Raised when agent execution fails."""

    pass


class ToolExecutionError(TaskFlowError):
    """Raised when tool execution fails."""

    pass


class LLMError(TaskFlowError):
    """Raised when LLM API call fails."""

    pass


class ConfigurationError(TaskFlowError):
    """Raised when configuration is invalid."""

    pass


# ADD THESE NEW EXCEPTIONS:
class CircuitBreakerOpenError(TaskFlowError):
    """Raised when circuit breaker is open (too many failures)."""

    pass


class RetryExhaustedError(TaskFlowError):
    """Raised when retry attempts are exhausted."""

    pass


class TimeoutError(TaskFlowError):
    """Raised when operation times out."""

    pass

class InvalidStateError(TaskFlowError):
    """Raised when agent/tool is in invalid state."""

    pass

class MaxRetriesExceededError(TaskFlowError):
    """Raised when maximum retry attempts exceeded."""

    pass
