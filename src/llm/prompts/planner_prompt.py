"""Prompt template for ReportPlannerAgent."""


def create_planner_prompt(tool_results, report_type: str, quality_assessment) -> str:
    """Create prompt for report planning.

    Args:
        tool_results: AnalysisOrchestratorOutput
        report_type: Type of report to generate
        quality_assessment: PrePromptEvaluatorOutput

    Returns:
        Formatted prompt string
    """

    # Extract key data
    total_comments = tool_results.total_comments
    sentiment_dist = tool_results.sentiment_distribution
    emotion_dist = tool_results.emotion_distribution
    top_topics = tool_results.top_topics[:5]  # Top 5 topics
    entities = tool_results.entities[:10]  # Top 10 entities
    keyphrases = tool_results.keyphrases[:10]  # Top 10 keyphrases

    prompt = f"""You are an expert data analyst creating a report structure for sentiment analysis results.

# Task
Create a detailed outline for a {report_type.upper()} report based on the analysis of {total_comments} customer comments.

# Analysis Summary

## Sentiment Distribution
{_format_distribution(sentiment_dist)}

## Emotion Distribution
{_format_distribution(emotion_dist)}

## Top Topics
{_format_topics(top_topics)}

## Key Entities Mentioned
{_format_entities(entities)}

## Important Keyphrases
{', '.join(keyphrases[:10])}

## Quality Score
Overall quality: {quality_assessment.quality_score:.1f}/100
Status: {quality_assessment.status}

# Report Type Guidelines

{_get_report_guidelines(report_type)}

# Your Task

Create a comprehensive report outline in JSON format with this structure:

{{
    "title": "Report title",
    "sections": [
        {{
            "name": "Section name",
            "priority": "high|medium|low",
            "key_points": ["Point 1", "Point 2"],
            "data_sources": ["sentiment", "topics", "entities"],
            "estimated_length": "2-3 paragraphs"
        }}
    ],
    "recommended_length": "5-7 pages",
    "key_insights": ["Insight 1", "Insight 2"],
    "recommended_focus": ["Focus area 1", "Focus area 2"],
    "narrative_flow": ["Flow step 1", "Flow step 2"]
}}

Important:
- Include 5-8 sections appropriate for {report_type} report
- Prioritize sections based on data quality and relevance
- Ensure logical narrative flow
- Focus on actionable insights
- Respond ONLY with valid JSON, no other text"""

    return prompt


def _format_distribution(dist: dict) -> str:
    """Format distribution dictionary."""
    total = sum(dist.values())
    lines = []
    for label, count in sorted(dist.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total * 100) if total > 0 else 0
        lines.append(f"- {label}: {count} ({percentage:.1f}%)")
    return "\n".join(lines)


def _format_topics(topics) -> str:
    """Format topics list."""
    lines = []
    for topic in topics:
        sentiment = (
            "positive"
            if topic.avg_sentiment > 0.3
            else "negative"
            if topic.avg_sentiment < -0.3
            else "neutral"
        )
        lines.append(
            f"- {topic.topic}: {topic.count} mentions ({sentiment} sentiment: {topic.avg_sentiment:.2f})"
        )
    return "\n".join(lines) if lines else "No significant topics detected"


def _format_entities(entities) -> str:
    """Format entities list."""
    lines = []
    for entity in entities:
        sentiment = (
            "positive"
            if entity.sentiment > 0.3
            else "negative"
            if entity.sentiment < -0.3
            else "neutral"
        )
        lines.append(f"- {entity.text} ({entity.type}): {entity.count} mentions ({sentiment})")
    return "\n".join(lines) if lines else "No entities detected"


def _get_report_guidelines(report_type: str) -> str:
    """Get guidelines for specific report type."""

    guidelines = {
        "executive": """
EXECUTIVE REPORT:
- Focus on high-level insights and key metrics
- Include executive summary (1-2 paragraphs)
- Highlight critical issues and opportunities
- Provide clear recommendations
- Keep sections concise and actionable
- Target length: 3-5 pages
""",
        "marketing": """
MARKETING REPORT:
- Analyze customer sentiment and emotions
- Identify campaign effectiveness
- Highlight brand perception
- Compare platforms/channels if applicable
- Focus on audience insights
- Include actionable marketing recommendations
- Target length: 5-7 pages
""",
        "product": """
PRODUCT REPORT:
- Focus on feature feedback and pain points
- Identify bugs and issues mentioned
- Analyze satisfaction with specific features
- Prioritize improvement areas
- Include user suggestions
- Provide product roadmap insights
- Target length: 7-10 pages
""",
        "customer_service": """
CUSTOMER SERVICE REPORT:
- Identify common support issues
- Analyze response effectiveness
- Highlight service quality trends
- Focus on resolution patterns
- Include training recommendations
- Provide escalation insights
- Target length: 5-8 pages
""",
        "comprehensive": """
COMPREHENSIVE REPORT:
- Cover all aspects of the analysis
- Deep dive into sentiment, emotions, topics
- Detailed entity analysis
- Cross-reference different data points
- Include visualizations suggestions
- Provide extensive recommendations
- Target length: 15-20 pages
""",
    }

    return guidelines.get(report_type, guidelines["executive"])
