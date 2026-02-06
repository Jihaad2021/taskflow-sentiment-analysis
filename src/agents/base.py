"""
Base Agent - Abstract base class for all TaskFlow agents.

This module defines the abstract base class that all agents in the TaskFlow
system must inherit from. It provides common functionality like logging,
metrics tracking, error handling, retry logic, and circuit breaker patterns.

All agents follow a consistent interface:
1. Receive Pydantic BaseModel input
2. Process/transform data
3. Return Pydantic BaseModel output

Example:
    >>> from pydantic import BaseModel
    >>> 
    >>> class MyInput(BaseModel):
    ...     text: str
    >>> 
    >>> class MyOutput(BaseModel):
    ...     result: str
    >>> 
    >>> class MyAgent(BaseAgent):
    ...     def execute(self, input_data: MyInput) -> MyOutput:
    ...         return MyOutput(result=input_data.text.upper())
    ...
    >>> config = AgentConfig(name="my_agent", version="1.0")
    >>> agent = MyAgent(config)
    >>> result = agent.run(MyInput(text="hello"))
"""

import time
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, TypeVar
from datetime import datetime
from contextlib import contextmanager

from pydantic import BaseModel, Field

from src.utils.logger import setup_logger
from src.utils.exceptions import (
    AgentExecutionError,
    CircuitBreakerOpenError,
    InvalidStateError,
    MaxRetriesExceededError
)

# Type variable for Pydantic models
InputT = TypeVar('InputT', bound=BaseModel)
OutputT = TypeVar('OutputT', bound=BaseModel)


class AgentConfig(BaseModel):
    """Configuration for BaseAgent.
    
    Attributes:
        name: Agent identifier (e.g., 'column_detector', 'data_validator')
        version: Agent version string (default: '1.0.0')
        log_level: Logging level (default: 'INFO')
        enable_circuit_breaker: Enable circuit breaker pattern (default: True)
        circuit_breaker_threshold: Failures before opening circuit (default: 5)
        circuit_breaker_timeout: Seconds before attempting reset (default: 60)
        enable_metrics: Enable metrics tracking (default: True)
    
    Example:
        >>> config = AgentConfig(
        ...     name="sentiment_agent",
        ...     version="1.0",
        ...     circuit_breaker_threshold=3
        ... )
    """
    name: str = Field(..., description="Agent name/identifier")
    version: str = Field(default="1.0.0", description="Agent version")
    log_level: str = Field(default="INFO", description="Logging level")
    enable_circuit_breaker: bool = Field(default=True, description="Enable circuit breaker")
    circuit_breaker_threshold: int = Field(default=5, ge=1, description="Failures before opening circuit")
    circuit_breaker_timeout: int = Field(default=60, ge=1, description="Seconds before reset attempt")
    enable_metrics: bool = Field(default=True, description="Enable metrics tracking")


class BaseAgent(ABC):
    """
    Abstract base class for all TaskFlow agents.
    
    Provides common functionality:
    - Standardized execute interface (Pydantic models)
    - Automatic metrics tracking
    - Integrated logging
    - Error handling with retry logic
    - Circuit breaker pattern
    - Performance monitoring
    - Input validation
    - Cost tracking (for LLM agents)
    
    All agents must implement the execute() method which takes a Pydantic
    BaseModel input and returns a Pydantic BaseModel output.
    
    Attributes:
        config: Agent configuration
        name: Agent identifier (from config)
        version: Agent version (from config)
        logger: Logger instance for this agent
        metrics: Dictionary tracking agent performance
        circuit_breaker: Circuit breaker state for failure handling
        
    Example:
        >>> class ColumnDetectorInput(BaseModel):
        ...     dataframe: Any
        >>> 
        >>> class ColumnDetectorOutput(BaseModel):
        ...     column_name: str
        ...     confidence: float
        >>> 
        >>> class ColumnDetectorAgent(BaseAgent):
        ...     def execute(self, input_data: ColumnDetectorInput) -> ColumnDetectorOutput:
        ...         # Detection logic
        ...         return ColumnDetectorOutput(column_name="comment", confidence=0.95)
        ...
        >>> config = AgentConfig(name="column_detector", version="1.0")
        >>> agent = ColumnDetectorAgent(config)
        >>> result = agent.run(input_data)
    """
    
    def __init__(self, config: AgentConfig):
        """
        Initialize base agent with configuration.
        
        Args:
            config: AgentConfig instance with agent settings
        
        Example:
            >>> config = AgentConfig(name="my_agent", version="2.0")
            >>> agent = MyAgent(config)
        """
        self.config = config
        self.name = config.name
        self.version = config.version
        
        # Setup logger with agent-specific name
        self.logger = setup_logger(
            name=f"agent.{self.name}",
            level=config.log_level
        )
        
        # Initialize metrics
        if config.enable_metrics:
            self.metrics = {
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "retried_calls": 0,
                "total_time_seconds": 0.0,
                "average_time_seconds": 0.0,
                "min_time_seconds": float('inf'),
                "max_time_seconds": 0.0,
                "last_execution_time": None,
                "total_cost": 0.0,
                "created_at": datetime.now().isoformat()
            }
        else:
            self.metrics = None
        
        # Initialize circuit breaker
        if config.enable_circuit_breaker:
            self.circuit_breaker = {
                "enabled": True,
                "failures": 0,
                "threshold": config.circuit_breaker_threshold,
                "is_open": False,
                "opened_at": None,
                "timeout": config.circuit_breaker_timeout
            }
        else:
            self.circuit_breaker = {"enabled": False}
        
        self.log(f"Agent '{self.name}' (v{self.version}) initialized", level="debug")
    
    @abstractmethod
    def execute(self, input_data: BaseModel) -> BaseModel:
        """
        Execute agent logic and return output.
        
        This is the main method that must be implemented by all agents.
        It receives Pydantic input model, processes it, and returns
        Pydantic output model.
        
        Args:
            input_data: Pydantic input model (e.g., ColumnDetectorInput)
        
        Returns:
            Pydantic output model (e.g., ColumnDetectorOutput)
        
        Raises:
            AgentExecutionError: If execution fails
        
        Note:
            This method is abstract and must be implemented by subclasses.
        
        Example:
            >>> def execute(self, input_data: MyInput) -> MyOutput:
            ...     # Your agent logic here
            ...     result = self.process(input_data.text)
            ...     return MyOutput(result=result)
        """
        pass
    
    async def execute_async(self, input_data: BaseModel) -> BaseModel:
        """
        Async version of execute for agents that need it.
        
        Default implementation calls sync execute in thread pool.
        Override for true async operations (e.g., LLM API calls).
        
        Args:
            input_data: Pydantic input model
        
        Returns:
            Pydantic output model
        
        Example:
            >>> async def execute_async(self, input_data: MyInput) -> MyOutput:
            ...     # Async LLM call
            ...     result = await self.llm.generate(input_data.prompt)
            ...     return MyOutput(result=result)
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.execute, input_data)
    
    def validate_input(self, input_data: BaseModel) -> bool:
        """
        Validate input data before execution.
        
        Override in subclass for custom validation logic.
        Default implementation uses Pydantic validation.
        
        Args:
            input_data: Pydantic model to validate
        
        Returns:
            True if valid, False otherwise
        
        Example:
            >>> def validate_input(self, input_data: ColumnDetectorInput) -> bool:
            ...     if input_data.dataframe is None:
            ...         return False
            ...     if len(input_data.dataframe) == 0:
            ...         return False
            ...     return True
        """
        # Pydantic already validates structure
        # This is for additional business logic validation
        return isinstance(input_data, BaseModel)
    
    def estimate_cost(self, input_data: BaseModel) -> float:
        """
        Estimate cost for this execution.
        
        Override in LLM agents to calculate token-based costs.
        Default returns 0.0 for non-LLM agents.
        
        Args:
            input_data: Input to estimate cost for
        
        Returns:
            Estimated cost in USD
        
        Example:
            >>> def estimate_cost(self, input_data: ReportWriterInput) -> float:
            ...     prompt_tokens = self.count_tokens(input_data.prompt)
            ...     estimated_output = 2000
            ...     return (prompt_tokens * 0.003 + estimated_output * 0.015) / 1000
        """
        return 0.0
    
    def run(
        self,
        input_data: BaseModel,
        max_retries: int = 0,
        retry_delay: float = 1.0,
        exponential_backoff: bool = True
    ) -> BaseModel:
        """
        Execute agent with automatic tracking, error handling, and retry logic.
        
        This is a wrapper around execute() that handles:
        - Input validation
        - Circuit breaker checks
        - Timing measurement
        - Retry logic with exponential backoff
        - Metrics updates
        - Error handling
        - Logging
        
        Users should call run() instead of execute() directly.
        
        Args:
            input_data: Pydantic input model
            max_retries: Maximum retry attempts (default: 0)
            retry_delay: Initial delay between retries in seconds (default: 1.0)
            exponential_backoff: Use exponential backoff for retries (default: True)
        
        Returns:
            Pydantic output model
        
        Raises:
            InvalidStateError: If input validation fails
            CircuitBreakerOpenError: If circuit breaker is open
            MaxRetriesExceededError: If max retries exceeded
            AgentExecutionError: If execution fails after retries
        
        Example:
            >>> # Simple execution
            >>> result = agent.run(input_data)
            >>>
            >>> # With retries (for LLM agents)
            >>> result = agent.run(input_data, max_retries=3, retry_delay=2.0)
        """
        # Validate input
        if not self.validate_input(input_data):
            raise InvalidStateError(
                agent_name=self.name,
                message="Input validation failed",
                details={"input_type": type(input_data).__name__}
            )
        
        # Check circuit breaker
        self._check_circuit_breaker()
        
        # Try execution with retries
        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                # Calculate delay for this attempt
                if attempt > 0:
                    delay = self._calculate_retry_delay(
                        attempt,
                        retry_delay,
                        exponential_backoff
                    )
                    self.log(
                        f"Retry {attempt}/{max_retries} after {delay:.2f}s delay",
                        level="warning"
                    )
                    time.sleep(delay)
                    if self.metrics:
                        self.metrics["retried_calls"] += 1
                
                # Execute with tracking
                result = self._execute_with_tracking(input_data)
                
                # Reset circuit breaker on success
                self._on_success()
                
                return result
                
            except Exception as e:
                last_exception = e
                self.log(
                    f"Attempt {attempt + 1} failed: {str(e)}",
                    level="error" if attempt == max_retries else "warning"
                )
                
                # Record failure
                self._on_failure()
                
                # If this was the last attempt, raise
                if attempt == max_retries:
                    break
        
        # All retries exhausted
        if max_retries > 0:
            raise MaxRetriesExceededError(
                agent_name=self.name,
                max_retries=max_retries,
                last_error=str(last_exception)
            ) from last_exception
        else:
            # Wrap in AgentExecutionError if not already
            if isinstance(last_exception, AgentExecutionError):
                raise last_exception
            else:
                raise AgentExecutionError(
                    agent_name=self.name,
                    message=str(last_exception),
                    details={
                        "input_type": type(input_data).__name__,
                        "original_error": type(last_exception).__name__
                    }
                ) from last_exception
    
    async def run_async(
        self,
        input_data: BaseModel,
        max_retries: int = 0,
        retry_delay: float = 1.0,
        exponential_backoff: bool = True
    ) -> BaseModel:
        """
        Async version of run() for concurrent execution.
        
        Args:
            input_data: Pydantic input model
            max_retries: Maximum retry attempts (default: 0)
            retry_delay: Initial delay between retries in seconds (default: 1.0)
            exponential_backoff: Use exponential backoff for retries (default: True)
        
        Returns:
            Pydantic output model
        
        Example:
            >>> # Parallel tool execution
            >>> results = await asyncio.gather(
            ...     sentiment_tool.run_async(input1),
            ...     emotion_tool.run_async(input2),
            ...     topic_tool.run_async(input3)
            ... )
        """
        # Validate input
        if not self.validate_input(input_data):
            raise InvalidStateError(
                agent_name=self.name,
                message="Input validation failed",
                details={"input_type": type(input_data).__name__}
            )
        
        # Check circuit breaker
        self._check_circuit_breaker()
        
        # Try execution with retries
        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    delay = self._calculate_retry_delay(
                        attempt,
                        retry_delay,
                        exponential_backoff
                    )
                    self.log(
                        f"Async retry {attempt}/{max_retries} after {delay:.2f}s",
                        level="warning"
                    )
                    await asyncio.sleep(delay)
                    if self.metrics:
                        self.metrics["retried_calls"] += 1
                
                # Execute async with tracking
                result = await self._execute_async_with_tracking(input_data)
                
                # Reset circuit breaker on success
                self._on_success()
                
                return result
                
            except Exception as e:
                last_exception = e
                self.log(
                    f"Async attempt {attempt + 1} failed: {str(e)}",
                    level="error" if attempt == max_retries else "warning"
                )
                
                self._on_failure()
                
                if attempt == max_retries:
                    break
        
        # All retries exhausted
        if max_retries > 0:
            raise MaxRetriesExceededError(
                agent_name=self.name,
                max_retries=max_retries,
                last_error=str(last_exception)
            ) from last_exception
        else:
            if isinstance(last_exception, AgentExecutionError):
                raise last_exception
            else:
                raise AgentExecutionError(
                    agent_name=self.name,
                    message=str(last_exception),
                    details={
                        "input_type": type(input_data).__name__,
                        "original_error": type(last_exception).__name__
                    }
                ) from last_exception
    
    def _execute_with_tracking(self, input_data: BaseModel) -> BaseModel:
        """Execute agent with timing and cost tracking (internal)."""
        start_time = time.time()
        
        # Log execution start
        self.log(
            f"Executing {self.name} with input type {type(input_data).__name__}",
            level="info"
        )
        
        # Estimate and log cost
        estimated_cost = self.estimate_cost(input_data)
        if estimated_cost > 0:
            self.log(f"Estimated cost: ${estimated_cost:.4f}", level="debug")
        
        # Execute agent logic
        output = self.execute(input_data)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Update metrics
        if self.metrics:
            self._update_metrics(
                success=True,
                execution_time=execution_time,
                cost=estimated_cost
            )
        
        # Log execution (compatible with old log_execution)
        self.log_execution(input_data, output, execution_time, estimated_cost)
        
        return output
    
    async def _execute_async_with_tracking(self, input_data: BaseModel) -> BaseModel:
        """Async version of _execute_with_tracking (internal)."""
        start_time = time.time()
        
        self.log(
            f"Executing {self.name} async with input type {type(input_data).__name__}",
            level="info"
        )
        
        estimated_cost = self.estimate_cost(input_data)
        if estimated_cost > 0:
            self.log(f"Estimated cost: ${estimated_cost:.4f}", level="debug")
        
        output = await self.execute_async(input_data)
        
        execution_time = time.time() - start_time
        
        if self.metrics:
            self._update_metrics(
                success=True,
                execution_time=execution_time,
                cost=estimated_cost
            )
        
        self.log_execution(input_data, output, execution_time, estimated_cost)
        
        return output
    
    def log_execution(
        self,
        input_data: BaseModel,
        output: BaseModel,
        execution_time: float = 0.0,
        cost: float = 0.0
    ) -> None:
        """
        Log execution details (backward compatible).
        
        Args:
            input_data: Input that was processed
            output: Result that was produced
            execution_time: Time taken in seconds
            cost: Cost in USD
        """
        self.logger.info(
            "Agent executed successfully",
            extra={
                "agent": self.name,
                "input_type": type(input_data).__name__,
                "output_type": type(output).__name__,
                "execution_time": execution_time,
                "cost": cost
            }
        )
    
    def _calculate_retry_delay(
        self,
        attempt: int,
        base_delay: float,
        exponential: bool
    ) -> float:
        """Calculate delay for retry attempt."""
        if exponential:
            return base_delay * (2 ** (attempt - 1))
        else:
            return base_delay * attempt
    
    def _check_circuit_breaker(self) -> None:
        """Check if circuit breaker is open."""
        if not self.circuit_breaker.get("enabled"):
            return
        
        if not self.circuit_breaker.get("is_open"):
            return
        
        # Check if timeout has elapsed
        opened_at = self.circuit_breaker.get("opened_at")
        timeout = self.circuit_breaker.get("timeout")
        
        if opened_at and (time.time() - opened_at) > timeout:
            self.log("Circuit breaker timeout elapsed, attempting reset", level="info")
            self.circuit_breaker["is_open"] = False
            self.circuit_breaker["failures"] = 0
            self.circuit_breaker["opened_at"] = None
            return
        
        # Circuit still open
        raise CircuitBreakerOpenError(
            agent_name=self.name,
            failures=self.circuit_breaker["failures"],
            threshold=self.circuit_breaker["threshold"],
            timeout=self.circuit_breaker["timeout"]
        )
    
    def _on_success(self) -> None:
        """Handle successful execution."""
        if not self.circuit_breaker.get("enabled"):
            return
        
        if self.circuit_breaker.get("failures", 0) > 0:
            self.log("Circuit breaker: Resetting failure count", level="debug")
            self.circuit_breaker["failures"] = 0
    
    def _on_failure(self) -> None:
        """Handle failed execution."""
        if not self.circuit_breaker.get("enabled"):
            return
        
        self.circuit_breaker["failures"] = self.circuit_breaker.get("failures", 0) + 1
        
        if self.circuit_breaker["failures"] >= self.circuit_breaker["threshold"]:
            self.circuit_breaker["is_open"] = True
            self.circuit_breaker["opened_at"] = time.time()
            self.log(
                f"Circuit breaker OPENED after {self.circuit_breaker['failures']} failures",
                level="error"
            )
    
    def log(self, message: str, level: str = "info") -> None:
        """Log message with agent context."""
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(f"[{self.name}] {message}")
    
    def _update_metrics(
        self,
        success: bool,
        execution_time: float,
        cost: float = 0.0
    ) -> None:
        """Update agent performance metrics (internal)."""
        if not self.metrics:
            return
        
        self.metrics["total_calls"] += 1
        
        if success:
            self.metrics["successful_calls"] += 1
        else:
            self.metrics["failed_calls"] += 1
        
        # Update timing metrics
        self.metrics["total_time_seconds"] += execution_time
        self.metrics["average_time_seconds"] = (
            self.metrics["total_time_seconds"] / self.metrics["total_calls"]
        )
        self.metrics["last_execution_time"] = execution_time
        
        # Update min/max times
        if execution_time < self.metrics["min_time_seconds"]:
            self.metrics["min_time_seconds"] = execution_time
        if execution_time > self.metrics["max_time_seconds"]:
            self.metrics["max_time_seconds"] = execution_time
        
        # Update cost
        self.metrics["total_cost"] += cost
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current agent metrics."""
        if not self.metrics:
            return {
                "agent_name": self.name,
                "metrics_enabled": False
            }
        
        success_rate = 0.0
        if self.metrics["total_calls"] > 0:
            success_rate = (
                self.metrics["successful_calls"] / self.metrics["total_calls"]
            ) * 100
        
        average_cost = 0.0
        if self.metrics["total_calls"] > 0:
            average_cost = self.metrics["total_cost"] / self.metrics["total_calls"]
        
        min_time = self.metrics["min_time_seconds"]
        if min_time == float('inf'):
            min_time = 0.0
        
        return {
            "agent_name": self.name,
            "agent_version": self.version,
            "total_calls": self.metrics["total_calls"],
            "successful_calls": self.metrics["successful_calls"],
            "failed_calls": self.metrics["failed_calls"],
            "retried_calls": self.metrics["retried_calls"],
            "success_rate": round(success_rate, 2),
            "total_time_seconds": round(self.metrics["total_time_seconds"], 2),
            "average_time_seconds": round(self.metrics["average_time_seconds"], 3),
            "min_time_seconds": round(min_time, 3),
            "max_time_seconds": round(self.metrics["max_time_seconds"], 3),
            "last_execution_time": self.metrics["last_execution_time"],
            "total_cost": round(self.metrics["total_cost"], 4),
            "average_cost": round(average_cost, 6),
            "circuit_breaker_state": {
                "enabled": self.circuit_breaker.get("enabled", False),
                "is_open": self.circuit_breaker.get("is_open", False),
                "failures": self.circuit_breaker.get("failures", 0),
                "threshold": self.circuit_breaker.get("threshold", 0)
            },
            "created_at": self.metrics["created_at"]
        }
    
    def reset_metrics(self) -> None:
        """Reset all metrics to initial state."""
        if not self.metrics:
            return
        
        created_at = self.metrics["created_at"]
        
        self.metrics = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "retried_calls": 0,
            "total_time_seconds": 0.0,
            "average_time_seconds": 0.0,
            "min_time_seconds": float('inf'),
            "max_time_seconds": 0.0,
            "last_execution_time": None,
            "total_cost": 0.0,
            "created_at": created_at
        }
        
        self.log("Metrics reset", level="debug")
    
    def reset_circuit_breaker(self) -> None:
        """Manually reset circuit breaker."""
        if not self.circuit_breaker.get("enabled"):
            return
        
        self.circuit_breaker["failures"] = 0
        self.circuit_breaker["is_open"] = False
        self.circuit_breaker["opened_at"] = None
        self.log("Circuit breaker manually reset", level="info")
    
    def get_info(self) -> Dict[str, Any]:
        """Get agent information."""
        return {
            "name": self.name,
            "version": self.version,
            "class": self.__class__.__name__,
            "circuit_breaker_enabled": self.circuit_breaker.get("enabled", False),
            "metrics_enabled": self.metrics is not None
        }
    
    @contextmanager
    def measure_time(self, operation: str = "operation"):
        """Context manager for measuring operation time."""
        start = time.time()
        self.log(f"Starting {operation}", level="debug")
        try:
            yield
        finally:
            elapsed = time.time() - start
            self.log(f"Completed {operation} in {elapsed:.3f}s", level="debug")
    
    def __enter__(self):
        """Context manager entry."""
        self.log("Agent context started", level="debug")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is not None:
            self.log(
                f"Agent context ended with error: {exc_val}",
                level="error"
            )
        else:
            self.log("Agent context ended successfully", level="debug")
        return False
    
    def __repr__(self) -> str:
        """String representation."""
        if self.metrics:
            metrics = self.get_metrics()
            return (
                f"{self.__class__.__name__}("
                f"name='{self.name}', "
                f"version='{self.version}', "
                f"calls={metrics['total_calls']}, "
                f"success_rate={metrics['success_rate']:.1f}%)"
            )
        else:
            return (
                f"{self.__class__.__name__}("
                f"name='{self.name}', "
                f"version='{self.version}')"
            )
    
    def __str__(self) -> str:
        """User-friendly string representation."""
        return f"{self.name} (v{self.version})"