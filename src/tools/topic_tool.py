"""Topic extraction tool using Sentence-BERT + clustering."""

from typing import Dict

from transformers import AutoModel, AutoTokenizer

from src.tools.base import BaseTool


class TopicTool(BaseTool):
    """Extract topics using sentence embeddings + TF-IDF."""

    def _load_model(self):
        """Load Sentence-BERT model."""
        self.logger.info(f"Loading topic model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name)

        # Move to device
        model = model.to(self.device)
        model.eval()

        self.logger.info("Topic model loaded successfully")
        return model

    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling for sentence embeddings."""
        import torch

        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def _get_embedding(self, text: str):
        """Get sentence embedding."""
        import torch

        # Tokenize
        encoded_input = self.tokenizer(
            text, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )

        # Move to device
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

        # Compute embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Mean pooling
        sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"])

        return sentence_embeddings.cpu().numpy()[0]

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
        stop_words = {"this", "that", "with", "from", "have", "been", "were", "they", "their"}

        keywords = [
            word.strip(".,!?;:") for word in words if len(word) > 4 and word not in stop_words
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
