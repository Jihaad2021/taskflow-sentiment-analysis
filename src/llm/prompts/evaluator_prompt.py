"""Prompt template for ReportEvaluatorAgent."""


def create_evaluator_prompt(
    report_text: str,
    original_data,
    report_plan,
) -> str:
    """Create prompt for report evaluation.

    Args:
        report_text: Generated report text
        original_data: AnalysisOrchestratorOutput
        report_plan: ReportPlannerOutput

    Returns:
        Formatted prompt string
    """

    outline = report_plan.report_outline

    prompt = f"""You are an expert report quality evaluator. Your task is to evaluate a sentiment analysis report for quality, accuracy, and completeness.

# Report to Evaluate

{report_text}

# Original Data Summary

- Total Comments: {original_data.total_comments}
- Sentiment Distribution: {_format_dist(original_data.sentiment_distribution)}
- Emotion Distribution: {_format_dist(original_data.emotion_distribution)}
- Top Topics: {len(original_data.top_topics)}
- Entities Found: {len(original_data.entities)}

# Expected Report Structure

**Title:** {outline.title}
**Expected Sections:** {len(outline.sections)}
**Required Sections:**
{_format_expected_sections(outline.sections)}

# Evaluation Criteria

Evaluate the report on these dimensions (score each 0-100):

1. **Completeness** (30% weight)
   - All required sections present?
   - Adequate coverage of each section?
   - No missing critical information?

2. **Factual Accuracy** (25% weight)
   - Numbers match the data?
   - Correct sentiment percentages?
   - Accurate topic counts?
   - No fabricated statistics?

3. **Coherence** (20% weight)
   - Logical flow between sections?
   - Clear narrative structure?
   - Consistent tone throughout?

4. **Actionability** (15% weight)
   - Clear recommendations provided?
   - Insights are practical?
   - Next steps identified?

5. **Hallucination Check** (10% weight)
   - No invented data?
   - No false claims?
   - All statements backed by data?

# Your Task

Evaluate the report and respond with JSON in this exact format:

{{
    "overall_score": 85.5,
    "should_regenerate": false,
    "checks": {{
        "completeness": {{
            "score": 90.0,
            "issues": ["Missing subsection X"],
            "critical": false
        }},
        "factual_accuracy": {{
            "score": 95.0,
            "issues": [],
            "critical": false
        }},
        "coherence": {{
            "score": 85.0,
            "issues": ["Transition between sections unclear"],
            "critical": false
        }},
        "actionability": {{
            "score": 80.0,
            "issues": ["Recommendations too vague"],
            "critical": false
        }},
        "hallucination": {{
            "score": 100.0,
            "issues": [],
            "critical": false
        }}
    }},
    "issues": ["Critical issue 1", "Critical issue 2"],
    "feedback": "Detailed feedback for regeneration if needed"
}}

**Decision Rules:**
- Set `should_regenerate: true` if overall_score < 70 OR any critical issue found
- Set `should_regenerate: false` if overall_score >= 70 AND no critical issues
- Mark issue as `critical: true` if it's a factual error or major omission
- In `feedback`, provide specific, actionable improvements

Respond ONLY with valid JSON."""

    return prompt


def _format_dist(dist: dict) -> str:
    """Format distribution."""
    items = []
    for label, count in sorted(dist.items(), key=lambda x: x[1], reverse=True):
        items.append(f"{label}={count}")
    return ", ".join(items)


def _format_expected_sections(sections) -> str:
    """Format expected sections."""
    lines = []
    for i, section in enumerate(sections, 1):
        lines.append(f"{i}. {section.name} (priority: {section.priority})")
    return "\n".join(lines)
