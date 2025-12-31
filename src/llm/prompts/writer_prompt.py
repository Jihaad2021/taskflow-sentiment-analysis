"""Prompt template for ReportWriterAgent."""

from typing import Optional


def create_writer_prompt(
    report_plan, tool_results, regeneration_feedback: Optional[str] = None
) -> str:
    """Create prompt for report writing.

    Args:
        report_plan: ReportPlannerOutput
        tool_results: AnalysisOrchestratorOutput
        regeneration_feedback: Optional feedback from previous generation

    Returns:
        Formatted prompt string
    """

    outline = report_plan.report_outline

    prompt = f"""You are an expert report writer specializing in data analysis and sentiment analysis reports.

# Task
Write a complete, professional report in Markdown format based on the provided outline and data.

# Report Outline

**Title:** {outline.title}
**Recommended Length:** {outline.recommended_length}

**Sections to Write:**
{_format_sections(outline.sections)}

# Data Summary

## Overview
- Total Comments Analyzed: {tool_results.total_comments}
- Sentiment Distribution: {_format_dist(tool_results.sentiment_distribution)}
- Emotion Distribution: {_format_dist(tool_results.emotion_distribution)}

## Key Topics
{_format_topics_for_writer(tool_results.top_topics[:5])}

## Entities Mentioned
{_format_entities_for_writer(tool_results.entities[:10])}

## Important Keyphrases
{', '.join(tool_results.keyphrases[:15])}

## Key Insights
{_format_list(report_plan.key_insights)}

## Recommended Focus Areas
{_format_list(report_plan.recommended_focus)}

# Writing Guidelines

1. **Format:** Write in Markdown with proper headers (##, ###)
2. **Tone:** Professional, data-driven, actionable
3. **Structure:** Follow the outline sections exactly
4. **Length:** Aim for {outline.recommended_length}
5. **Data:** Reference specific numbers and percentages
6. **Insights:** Provide actionable recommendations
7. **Flow:** Ensure smooth narrative between sections

# Important Rules

- Use Markdown formatting (headers, lists, bold, italic)
- Include specific metrics and percentages
- Provide concrete examples from the data
- End with clear, actionable recommendations
- DO NOT use tables (use lists instead)
- DO NOT include images or charts (describe them instead)

{_add_regeneration_feedback(regeneration_feedback)}

# Your Task

Write the complete report now. Start with "# {outline.title}" and include all sections."""

    return prompt


def _format_sections(sections) -> str:
    """Format sections for prompt."""
    lines = []
    for i, section in enumerate(sections, 1):
        lines.append(f"{i}. **{section.name}** (Priority: {section.priority})")
        lines.append(f"   Key points: {', '.join(section.key_points[:3])}")
        lines.append(f"   Length: {section.estimated_length}")
        lines.append("")
    return "\n".join(lines)


def _format_dist(dist: dict) -> str:
    """Format distribution."""
    total = sum(dist.values())
    items = []
    for label, count in sorted(dist.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total * 100) if total > 0 else 0
        items.append(f"{label} {count} ({pct:.1f}%)")
    return ", ".join(items)


def _format_topics_for_writer(topics) -> str:
    """Format topics for writer."""
    if not topics:
        return "No significant topics detected"

    lines = []
    for topic in topics:
        sentiment_label = (
            "positive"
            if topic.avg_sentiment > 0.3
            else "negative"
            if topic.avg_sentiment < -0.3
            else "neutral"
        )
        lines.append(
            f"- **{topic.topic}**: {topic.count} mentions "
            f"({sentiment_label}, score: {topic.avg_sentiment:.2f})"
        )
    return "\n".join(lines)


def _format_entities_for_writer(entities) -> str:
    """Format entities for writer."""
    if not entities:
        return "No entities detected"

    lines = []
    for entity in entities[:10]:
        sentiment_label = (
            "positive"
            if entity.sentiment > 0.3
            else "negative"
            if entity.sentiment < -0.3
            else "neutral"
        )
        lines.append(
            f"- **{entity.text}** ({entity.type}): {entity.count} mentions ({sentiment_label})"
        )
    return "\n".join(lines)


def _format_list(items) -> str:
    """Format list of items."""
    if not items:
        return "None specified"
    return "\n".join([f"- {item}" for item in items])


def _add_regeneration_feedback(feedback: Optional[str] = None) -> str:
    """Add regeneration feedback if present."""
    if not feedback:
        return ""

    return f"""
# IMPORTANT: Previous Attempt Feedback

The previous version of this report had issues. Please address these:

{feedback}

Make sure to improve these specific areas in your new version.
"""
