"""Keyphrase extraction tool using DistilBERT."""

from typing import Dict

from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    TokenClassificationPipeline,
)

from src.tools.base import BaseTool


class KeyphraseTool(BaseTool):
    """Extract keyphrases using keyphrase-extraction-distilbert."""

    def _load_model(self):
        """Load keyphrase extraction model."""
        self.logger.info(f"Loading keyphrase model: {self.model_name}")

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForTokenClassification.from_pretrained(self.model_name)

        # Create pipeline
        extractor = TokenClassificationPipeline(
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=0 if self.device == "cuda" else -1,
        )

        self.logger.info("Keyphrase model loaded successfully")
        return extractor

    def analyze(self, text: str) -> Dict:
        """Extract keyphrases from text.

        Args:
            text: Input text

        Returns:
            Dictionary with keyphrase results
        """
        # Extract keyphrases
        results = self.model(text)

        # Format keyphrases with scores
        keyphrases = []
        scores = {}

        for item in results:
            phrase = item["word"].strip()
            score = float(item["score"])

            if phrase and len(phrase) > 2:  # Filter very short phrases
                if phrase not in keyphrases:
                    keyphrases.append(phrase)
                    scores[phrase] = score

        # Sort by score
        keyphrases = sorted(keyphrases, key=lambda x: scores[x], reverse=True)

        return {"keyphrases": keyphrases, "scores": scores}
