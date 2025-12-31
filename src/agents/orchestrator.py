"""Analysis orchestrator agent - coordinates tool execution."""

import time
from collections import Counter, defaultdict
from typing import Any, Dict, List

from pydantic import BaseModel

from src.agents.base import BaseAgent
from src.models.schemas import (
    AnalysisOrchestratorInput,
    AnalysisOrchestratorOutput,
    CommentAnalysis,
    EmotionResult,
    Entity,
    EntityResult,
    EntitySummary,
    KeyphraseResult,
    SentimentResult,
    TopicResult,
    TopicSummary,
)
from src.tools.emotion_tool import EmotionTool
from src.tools.entity_tool import EntityTool
from src.tools.keyphrase_tool import KeyphraseTool
from src.tools.sentiment_tool import SentimentTool
from src.tools.topic_tool import TopicTool
from src.utils.exceptions import AgentExecutionError


class AnalysisOrchestratorAgent(BaseAgent):
    """Orchestrate execution of 5 analysis tools."""

    def __init__(self, device: str = "cpu"):
        """Initialize orchestrator.

        Args:
            device: Device for models ('cpu' or 'cuda')
        """
        from src.models.schemas import AgentConfig

        super().__init__(AgentConfig(name="AnalysisOrchestratorAgent"))

        self.device = device

        # Initialize tools
        self.logger.info("Initializing analysis tools...")
        self.sentiment_tool = SentimentTool(
            "cardiffnlp/twitter-roberta-base-sentiment-latest", device=device
        )
        self.emotion_tool = EmotionTool(
            "j-hartmann/emotion-english-distilroberta-base", device=device
        )
        self.topic_tool = TopicTool("sentence-transformers/all-MiniLM-L6-v2", device=device)
        self.entity_tool = EntityTool("dslim/bert-base-NER", device=device)
        self.keyphrase_tool = KeyphraseTool(
            "ml6team/keyphrase-extraction-distilbert-inspec", device=device
        )
        self.logger.info("All tools initialized successfully")

    def execute(self, input_data: BaseModel) -> BaseModel:
        """Execute all tools on comments.

        Args:
            input_data: AnalysisOrchestratorInput

        Returns:
            AnalysisOrchestratorOutput with aggregated results
        """
        if not isinstance(input_data, AnalysisOrchestratorInput):
            raise AgentExecutionError("Invalid input type")

        start_time = time.time()

        comments = input_data.comments
        self.logger.info(f"Processing {len(comments)} comments")

        # Analyze all comments
        individual_results = self._analyze_all_comments(comments)

        # Aggregate results
        aggregated = self._aggregate_results(individual_results)

        execution_time = time.time() - start_time

        output = AnalysisOrchestratorOutput(
            total_comments=len(comments),
            sentiment_distribution=aggregated["sentiment_dist"],
            emotion_distribution=aggregated["emotion_dist"],
            top_topics=aggregated["top_topics"],
            entities=aggregated["entities"],
            keyphrases=aggregated["keyphrases"],
            individual_results=individual_results,
            execution_time=execution_time,
        )

        self.log_execution(input_data, output)
        self.logger.info(f"Processed {len(comments)} comments in {execution_time:.2f}s")

        return output

    def _analyze_all_comments(self, comments: List[str]) -> List[CommentAnalysis]:
        """Analyze all comments with all tools.

        Args:
            comments: List of comment texts

        Returns:
            List of CommentAnalysis objects
        """
        results = []

        for i, comment in enumerate(comments):
            start = time.time()

            # Run all 5 tools
            sentiment = self.sentiment_tool.analyze(comment)
            emotion = self.emotion_tool.analyze(comment)
            topics = self.topic_tool.analyze(comment)
            entities = self.entity_tool.analyze(comment)
            keyphrases = self.keyphrase_tool.analyze(comment)

            exec_time = time.time() - start

            # Create result object
            analysis = CommentAnalysis(
                comment_id=f"comment_{i}",
                text=comment,
                sentiment=SentimentResult(**sentiment),
                emotion=EmotionResult(**emotion),
                topics=TopicResult(**topics),
                entities=EntityResult(entities=[Entity(**e) for e in entities["entities"]]),
                keyphrases=KeyphraseResult(**keyphrases),
                execution_time=exec_time,
            )

            results.append(analysis)

        return results

    def _aggregate_results(self, results: List[CommentAnalysis]) -> Dict:
        """Aggregate individual results into summaries."""

        # Sentiment distribution
        sentiment_dist = Counter([r.sentiment.label for r in results])

        # Emotion distribution
        emotion_dist = Counter([r.emotion.emotion for r in results])

        # Topic aggregation - ADD TYPE ANNOTATION
        topic_data: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "sentiments": [], "samples": []}
        )
        for result in results:
            for topic in result.topics.topics[:3]:
                topic_data[topic]["count"] += 1
                topic_data[topic]["sentiments"].append(
                    self._sentiment_to_score(result.sentiment.label)
                )
                if len(topic_data[topic]["samples"]) < 5:
                    topic_data[topic]["samples"].append(result.text[:100])

        # Sort topics by count
        sorted_topics = sorted(
            topic_data.items(), key=lambda x: x[1]["count"], reverse=True  # type: ignore
        )[:10]

        top_topics = [
            TopicSummary(
                topic=topic,
                count=data["count"],  # type: ignore
                avg_sentiment=sum(data["sentiments"]) / len(data["sentiments"]),  # type: ignore
                sample_comments=data["samples"],  # type: ignore
            )
            for topic, data in sorted_topics
        ]

        # Entity aggregation - ADD TYPE ANNOTATION
        entity_data: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "sentiments": [], "contexts": []}
        )
        for result in results:
            for entity in result.entities.entities:
                key = f"{entity.text}|{entity.type}"
                entity_data[key]["count"] += 1
                entity_data[key]["sentiments"].append(
                    self._sentiment_to_score(result.sentiment.label)
                )
                if len(entity_data[key]["contexts"]) < 3:
                    entity_data[key]["contexts"].append(result.text[:100])

        # Sort entities by count
        sorted_entities = sorted(
            entity_data.items(), key=lambda x: x[1]["count"], reverse=True  # type: ignore
        )[:20]

        entities = [
            EntitySummary(
                text=key.split("|")[0],
                type=key.split("|")[1],
                count=data["count"],  # type: ignore
                sentiment=sum(data["sentiments"]) / len(data["sentiments"]),  # type: ignore
                contexts=data["contexts"],  # type: ignore
            )
            for key, data in sorted_entities
        ]

        # Keyphrase aggregation
        all_keyphrases = []
        for result in results:
            all_keyphrases.extend(result.keyphrases.keyphrases[:3])

        keyphrase_counts = Counter(all_keyphrases)
        top_keyphrases = [kp for kp, _ in keyphrase_counts.most_common(20)]

        return {
            "sentiment_dist": dict(sentiment_dist),
            "emotion_dist": dict(emotion_dist),
            "top_topics": top_topics,
            "entities": entities,
            "keyphrases": top_keyphrases,
        }

    def _sentiment_to_score(self, label: str) -> float:
        """Convert sentiment label to numeric score.

        Args:
            label: Sentiment label

        Returns:
            Score: positive=1, neutral=0, negative=-1
        """
        mapping = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
        return mapping.get(label, 0.0)
