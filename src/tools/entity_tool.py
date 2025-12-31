"""Named Entity Recognition tool using BERT."""

from typing import Dict

from transformers import pipeline

from src.tools.base import BaseTool


class EntityTool(BaseTool):
    """Extract named entities using BERT-NER."""

    def _load_model(self):
        """Load BERT NER model."""
        self.logger.info(f"Loading entity model: {self.model_name}")

        # Use pipeline for easier NER
        ner_pipeline = pipeline(
            "ner",
            model=self.model_name,
            tokenizer=self.model_name,
            aggregation_strategy="simple",  # Merge subword tokens
            device=0 if self.device == "cuda" else -1,
        )

        self.logger.info("Entity model loaded successfully")
        return ner_pipeline

    def analyze(self, text: str) -> Dict:
        """Extract named entities from text.

        Args:
            text: Input text

        Returns:
            Dictionary with entity results
        """
        # Run NER
        ner_results = self.model(text)

        # Format entities
        entities = []
        for entity in ner_results:
            entities.append(
                {
                    "text": entity["word"],
                    "type": entity["entity_group"],
                    "start": entity["start"],
                    "end": entity["end"],
                    "confidence": float(entity["score"]),
                }
            )

        return {"entities": entities}
