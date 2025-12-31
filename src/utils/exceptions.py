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
