"""Topic extraction tool using Sentence-BERT."""

from typing import Dict
from sentence_transformers import SentenceTransformer

from src.tools.base import BaseTool


class TopicTool(BaseTool):
    """Extract topics using sentence embeddings."""

    def _load_model(self):
        """Load Sentence-BERT model."""
        self.logger.info(f"Loading topic model: {self.model_name}")

        # Use SentenceTransformer instead of AutoModel
        model = SentenceTransformer(self.model_name)
        
        # Move to device (SentenceTransformer handles this differently)
        if self.device == "cuda":
            model = model.to(self.device)
        
        self.logger.info("Topic model loaded successfully")
        return model

    def analyze(self, text: str) -> Dict:
        """Extract topics from text.

        For single text, extract keywords using simple heuristics.

        Args:
            text: Input text

        Returns:
            Dictionary with topic results
        """
        # Simple topic extraction for single text
        # Extract nouns/keywords using basic heuristics

        words = text.lower().split()

        # Simple keyword extraction (words > 4 chars, not common words)
        stop_words = {
            "this", "that", "with", "from", "have", "been", "were", 
            "they", "their", "the", "and", "for", "are", "but"
        }

        keywords = [
            word.strip(".,!?;:\"'") 
            for word in words 
            if len(word) > 4 and word not in stop_words
        ]

        # Get unique keywords
        unique_keywords = list(dict.fromkeys(keywords))[:5]  # Top 5

        # Create topic labels
        topics = unique_keywords if unique_keywords else ["general"]

        # Assign relevance scores (simple: position-based)
        relevance_scores = {topic: 1.0 / (i + 1) for i, topic in enumerate(topics)}

        return {
            "topics": topics,
            "relevance_scores": relevance_scores,
            "primary_topic": topics[0] if topics else "general",
        }