"""Base tool class for all analysis tools."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from src.utils.exceptions import ToolExecutionError
from src.utils.logger import setup_logger


class BaseTool(ABC):
    """Base class for all analysis tools."""

    def __init__(self, model_name: str, device: str = "cpu"):
        """Initialize tool with model.

        Args:
            model_name: HuggingFace model identifier
            device: 'cpu' or 'cuda'
        """
        self.model_name = model_name
        self.device = device
        self.logger = setup_logger(self.__class__.__name__)
        self.model = self._load_model()

    @abstractmethod
    def _load_model(self) -> Any:
        """Load the model.

        Must be implemented by subclass.

        Returns:
            Loaded model instance
        """
        pass

    @abstractmethod
    def analyze(self, text: str) -> Dict:
        """Analyze single text.

        Must be implemented by subclass.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary with analysis results

        Raises:
            ToolExecutionError: If analysis fails
        """
        pass

    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """Analyze batch of texts.

        Default implementation processes sequentially.
        Override for optimized batch processing.

        Args:
            texts: List of texts to analyze

        Returns:
            List of result dictionaries
        """
        results = []
        for text in texts:
            try:
                result = self.analyze(text)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to analyze text: {e}")
                raise ToolExecutionError(f"Batch analysis failed: {e}") from e

        return results
