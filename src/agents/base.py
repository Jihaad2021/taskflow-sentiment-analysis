"""Base agent class for all agents."""

from abc import ABC, abstractmethod

from pydantic import BaseModel

from src.utils.logger import setup_logger


class BaseAgent(ABC):
    """Base class for all agents.

    All agents must inherit from this class and implement execute().
    """

    def __init__(self, config: BaseModel):
        """Initialize agent with configuration.

        Args:
            config: Pydantic configuration model
        """
        self.config = config
        self.logger = setup_logger(self.__class__.__name__)

    @abstractmethod
    def execute(self, input_data: BaseModel) -> BaseModel:
        """Execute agent logic.

        Must be implemented by all subclasses.

        Args:
            input_data: Pydantic input model

        Returns:
            Pydantic output model

        Raises:
            AgentExecutionError: If execution fails
        """
        pass

    def validate_input(self, input_data: BaseModel) -> bool:
        """Validate input data.

        Override in subclass for custom validation.

        Args:
            input_data: Input to validate

        Returns:
            True if valid, False otherwise
        """
        return True

    def log_execution(self, input_data: BaseModel, output: BaseModel) -> None:
        """Log execution details.

        Args:
            input_data: Input that was processed
            output: Result that was produced
        """
        self.logger.info(
            "Agent executed successfully",
            extra={
                "agent": self.__class__.__name__,
                "input_type": type(input_data).__name__,
                "output_type": type(output).__name__,
            },
        )
