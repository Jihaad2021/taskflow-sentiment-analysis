"""Emotion detection tool using DistilRoBERTa."""

from typing import Dict

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.tools.base import BaseTool


class EmotionTool(BaseTool):
    """Detect emotions using emotion-english-distilroberta-base."""

    def _load_model(self):
        """Load DistilRoBERTa emotion model."""
        self.logger.info(f"Loading emotion model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

        # Move to device
        model = model.to(self.device)
        model.eval()

        self.logger.info("Emotion model loaded successfully")
        return model

    def analyze(self, text: str) -> Dict:
        """Analyze emotion in text.

        Args:
            text: Input text

        Returns:
            Dictionary with emotion results
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

        # Model outputs 6 emotions
        labels = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
        label_scores = {labels[i]: float(scores[i]) for i in range(len(labels))}

        # Get top prediction
        top_idx = scores.argmax()
        top_emotion = labels[top_idx]
        top_confidence = float(scores[top_idx])

        return {"emotion": top_emotion, "confidence": top_confidence, "scores": label_scores}
