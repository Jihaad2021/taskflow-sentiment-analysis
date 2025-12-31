"""Pre-prompt evaluator agent - validate tool output quality."""

from typing import Any, Dict, List

from pydantic import BaseModel

from src.agents.base import BaseAgent
from src.models.schemas import (
    AnalysisOrchestratorOutput,
    PrePromptEvaluatorInput,
    PrePromptEvaluatorOutput,
    QualityCheck,
)
from src.utils.exceptions import AgentExecutionError


class PrePromptEvaluatorAgent(BaseAgent):
    """Evaluate quality of tool outputs before LLM processing."""

    def __init__(
        self,
        min_coverage: float = 0.8,
        min_confidence: float = 0.6,
        min_quality_score: float = 70.0,
    ):
        """Initialize evaluator.

        Args:
            min_coverage: Minimum data coverage (0-1)
            min_confidence: Minimum average confidence (0-1)
            min_quality_score: Minimum quality score to pass (0-100)
        """
        from src.models.schemas import AgentConfig

        super().__init__(AgentConfig(name="PrePromptEvaluatorAgent"))

        self.min_coverage = min_coverage
        self.min_confidence = min_confidence
        self.min_quality_score = min_quality_score

    def execute(self, input_data: BaseModel) -> BaseModel:
        """Evaluate tool output quality.

        Args:
            input_data: PrePromptEvaluatorInput

        Returns:
            PrePromptEvaluatorOutput with quality assessment
        """
        if not isinstance(input_data, PrePromptEvaluatorInput):
            raise AgentExecutionError("Invalid input type")

        tool_results = input_data.tool_results

        # Run quality checks
        checks = self._run_checks(tool_results)

        # Calculate overall score
        quality_score = self._calculate_quality_score(checks)

        # Determine status
        status, should_proceed = self._determine_status(quality_score, checks)

        # Generate issues and recommendations
        issues = [c.message for c in checks if not c.passed and c.score < 50]
        recommendations = self._generate_recommendations(checks)

        # Calculate metadata
        metadata = self._calculate_metadata(tool_results, checks)

        output = PrePromptEvaluatorOutput(
            status=status,
            quality_score=quality_score,
            checks=checks,
            issues=issues,
            recommendations=recommendations,
            should_proceed=should_proceed,
            metadata=metadata,
        )

        self.log_execution(input_data, output)
        self.logger.info(f"Quality score: {quality_score:.1f}/100, Status: {status}")

        return output

    def _run_checks(self, results: AnalysisOrchestratorOutput) -> List[QualityCheck]:
        """Run all quality checks.

        Args:
            results: Tool results to evaluate

        Returns:
            List of QualityCheck objects
        """
        checks = []

        # Check 1: Data coverage
        coverage_check = self._check_data_coverage(results)
        checks.append(coverage_check)

        # Check 2: Sentiment consistency
        sentiment_check = self._check_sentiment_quality(results)
        checks.append(sentiment_check)

        # Check 3: Topic quality
        topic_check = self._check_topic_quality(results)
        checks.append(topic_check)

        # Check 4: Entity extraction quality
        entity_check = self._check_entity_quality(results)
        checks.append(entity_check)

        # Check 5: Overall confidence
        confidence_check = self._check_confidence(results)
        checks.append(confidence_check)

        return checks

    def _check_data_coverage(self, results: AnalysisOrchestratorOutput) -> QualityCheck:
        """Check if enough data was successfully processed."""
        total = results.total_comments
        processed = len(results.individual_results)

        coverage = processed / total if total > 0 else 0
        score = coverage * 100

        passed = coverage >= self.min_coverage

        return QualityCheck(
            name="Data Coverage",
            passed=passed,
            score=score,
            message=f"Processed {processed}/{total} comments ({coverage*100:.1f}%)",
        )

    def _check_sentiment_quality(self, results: AnalysisOrchestratorOutput) -> QualityCheck:
        """Check sentiment distribution and consistency."""
        sentiment_dist = results.sentiment_distribution

        if not sentiment_dist:
            return QualityCheck(
                name="Sentiment Quality",
                passed=False,
                score=0.0,
                message="No sentiment data available",
            )

        total = sum(sentiment_dist.values())

        # Check for balanced distribution (not all same sentiment)
        max_ratio = max(sentiment_dist.values()) / total if total > 0 else 0

        # Score based on distribution diversity
        if max_ratio < 0.5:
            score = 100.0  # Very diverse
        elif max_ratio < 0.7:
            score = 80.0  # Good diversity
        elif max_ratio < 0.9:
            score = 60.0  # Acceptable
        else:
            score = 40.0  # Too uniform

        passed = score >= 60.0

        return QualityCheck(
            name="Sentiment Quality",
            passed=passed,
            score=score,
            message=f"Sentiment distribution diversity: {score:.0f}%",
        )

    def _check_topic_quality(self, results: AnalysisOrchestratorOutput) -> QualityCheck:
        """Check topic extraction quality."""
        topics = results.top_topics

        if not topics:
            return QualityCheck(
                name="Topic Quality", passed=False, score=0.0, message="No topics extracted"
            )

        # Check number of unique topics
        num_topics = len(topics)

        # Check topic coverage (how many comments per topic)
        avg_count = sum(t.count for t in topics) / len(topics) if topics else 0
        coverage_ratio = avg_count / results.total_comments if results.total_comments > 0 else 0

        # Score based on topic diversity and coverage
        if num_topics >= 5 and coverage_ratio > 0.3:
            score = 90.0
        elif num_topics >= 3 and coverage_ratio > 0.2:
            score = 75.0
        elif num_topics >= 2:
            score = 60.0
        else:
            score = 40.0

        passed = score >= 60.0

        return QualityCheck(
            name="Topic Quality",
            passed=passed,
            score=score,
            message=f"Extracted {num_topics} topics with {coverage_ratio*100:.1f}% coverage",
        )

    def _check_entity_quality(self, results: AnalysisOrchestratorOutput) -> QualityCheck:
        """Check entity extraction quality."""
        entities = results.entities

        num_entities = len(entities)

        # Score based on number of entities found
        if num_entities >= 10:
            score = 90.0
        elif num_entities >= 5:
            score = 75.0
        elif num_entities >= 2:
            score = 60.0
        elif num_entities >= 1:
            score = 50.0
        else:
            score = 30.0

        passed = score >= 50.0

        return QualityCheck(
            name="Entity Quality",
            passed=passed,
            score=score,
            message=f"Extracted {num_entities} unique entities",
        )

    def _check_confidence(self, results: AnalysisOrchestratorOutput) -> QualityCheck:
        """Check average confidence scores."""
        if not results.individual_results:
            return QualityCheck(
                name="Confidence Check", passed=False, score=0.0, message="No results to evaluate"
            )

        # Calculate average sentiment confidence
        sentiment_confidences = [r.sentiment.score for r in results.individual_results]
        avg_sentiment_conf = sum(sentiment_confidences) / len(sentiment_confidences)

        # Calculate average emotion confidence
        emotion_confidences = [r.emotion.confidence for r in results.individual_results]
        avg_emotion_conf = sum(emotion_confidences) / len(emotion_confidences)

        # Overall average
        avg_confidence = (avg_sentiment_conf + avg_emotion_conf) / 2

        score = avg_confidence * 100
        passed = avg_confidence >= self.min_confidence

        return QualityCheck(
            name="Confidence Check",
            passed=passed,
            score=score,
            message=f"Average confidence: {avg_confidence*100:.1f}%",
        )

    def _calculate_quality_score(self, checks: List[QualityCheck]) -> float:
        """Calculate overall quality score from checks.

        Args:
            checks: List of quality checks

        Returns:
            Overall score (0-100)
        """
        if not checks:
            return 0.0

        # Weighted average
        weights = {
            "Data Coverage": 0.3,
            "Confidence Check": 0.25,
            "Sentiment Quality": 0.2,
            "Topic Quality": 0.15,
            "Entity Quality": 0.1,
        }

        total_score = 0.0
        total_weight = 0.0

        for check in checks:
            weight = weights.get(check.name, 0.1)
            total_score += check.score * weight
            total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0

    def _determine_status(
        self, quality_score: float, checks: List[QualityCheck]
    ) -> tuple[str, bool]:
        """Determine overall status and whether to proceed.

        Args:
            quality_score: Overall quality score
            checks: List of quality checks

        Returns:
            Tuple of (status, should_proceed)
        """
        # Check for critical failures
        critical_checks = ["Data Coverage", "Confidence Check"]
        critical_failed = any(not c.passed for c in checks if c.name in critical_checks)

        if critical_failed:
            return "fail", False

        # Check quality score
        if quality_score >= self.min_quality_score:
            return "pass", True
        elif quality_score >= 60.0:
            return "warning", True
        else:
            return "fail", False

    def _generate_recommendations(self, checks: List[QualityCheck]) -> List[str]:
        """Generate recommendations based on check results.

        Args:
            checks: List of quality checks

        Returns:
            List of recommendation strings
        """
        recommendations = []

        for check in checks:
            if not check.passed:
                if check.name == "Data Coverage":
                    recommendations.append("Consider increasing data quality or quantity")
                elif check.name == "Sentiment Quality":
                    recommendations.append(
                        "Sentiment distribution is too uniform - verify data quality"
                    )
                elif check.name == "Topic Quality":
                    recommendations.append("Low topic diversity - consider more varied data")
                elif check.name == "Entity Quality":
                    recommendations.append("Few entities detected - data may lack specificity")
                elif check.name == "Confidence Check":
                    recommendations.append(
                        "Low confidence scores - verify text quality and clarity"
                    )

        return recommendations

    def _calculate_metadata(
        self, results: AnalysisOrchestratorOutput, checks: List[QualityCheck]
    ) -> Dict[str, Any]:
        """Calculate metadata statistics.

        Args:
            results: Tool results
            checks: Quality checks

        Returns:
            Metadata dictionary
        """
        # Get confidence check
        confidence_check = next((c for c in checks if c.name == "Confidence Check"), None)

        # Get coverage check
        coverage_check = next((c for c in checks if c.name == "Data Coverage"), None)

        return {
            "avg_confidence": confidence_check.score / 100 if confidence_check else 0.0,
            "data_coverage": coverage_check.score / 100 if coverage_check else 0.0,
            "num_topics": len(results.top_topics),
            "num_entities": len(results.entities),
            "num_keyphrases": len(results.keyphrases),
        }
