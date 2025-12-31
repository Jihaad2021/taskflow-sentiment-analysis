"""Sentiment analysis tool using RoBERTa."""

from typing import Dict

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.tools.base import BaseTool


class SentimentTool(BaseTool):
    """Analyze sentiment using twitter-roberta-base-sentiment."""

    def _load_model(self):
        """Load RoBERTa sentiment model."""
        self.logger.info(f"Loading sentiment model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

        # Move to device
        model = model.to(self.device)
        model.eval()

        self.logger.info("Sentiment model loaded successfully")
        return model

    def analyze(self, text: str) -> Dict:
        """Analyze sentiment of text.

        Args:
            text: Input text

        Returns:
            Dictionary with sentiment results
        """
        # Tokenize
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512, padding=True
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Predict
        import torch

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)

        # Get scores
        scores = probs[0].cpu().numpy()

        # Model outputs: [negative, neutral, positive]
        labels = ["negative", "neutral", "positive"]
        label_scores = {labels[i]: float(scores[i]) for i in range(len(labels))}

        # Get top prediction
        top_idx = scores.argmax()
        top_label = labels[top_idx]
        top_score = float(scores[top_idx])

        return {"label": top_label, "score": top_score, "scores": label_scores}
