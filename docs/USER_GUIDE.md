# TaskFlow User Guide

> **Complete guide for using TaskFlow Sentiment Analysis**

Welcome! This guide will help you analyze your customer feedback and generate professional reports in minutes.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Preparing Your Data](#preparing-your-data)
3. [Using the Web Interface](#using-the-web-interface)
4. [Understanding Your Report](#understanding-your-report)
5. [Report Types](#report-types)
6. [Best Practices](#best-practices)
7. [FAQ](#faq)

---

## Getting Started

### What You Need

✅ **CSV file** with customer comments/reviews  
✅ **At least 10 comments** (recommended: 50+)  
✅ **Internet connection** (for report generation)

### What You'll Get

📊 **Professional report** with:
- Sentiment analysis (positive/negative/neutral)
- Emotion detection (joy, anger, sadness, etc.)
- Key topics and themes
- Important entities (brands, products, features)
- Actionable recommendations

⏱️ **Time:** 30-120 seconds (depending on data size)  
💰 **Cost:** ~$0.015 per 1000 comments

---

## Preparing Your Data

### CSV Format

Your CSV file should have at least one column with text to analyze.

**✅ Good Example:**

```csv
review
Great product! Love the quality.
Terrible customer service.
Good value for money.
```

**✅ Also Works:**

```csv
review_id,customer_feedback,rating,date
1,Amazing! Highly recommend.,5,2024-01-01
2,Disappointed with quality.,2,2024-01-02
3,It's okay, nothing special.,3,2024-01-03
```

**✅ Any Column Name:**

```csv
comment,feedback,text,review,message
All of these work! System auto-detects.
```

### Data Requirements

| Requirement | Details |
|-------------|---------|
| **File Format** | CSV (.csv) |
| **File Size** | Maximum 10MB |
| **Minimum Rows** | 10 comments |
| **Maximum Rows** | 10,000 comments |
| **Text Length** | 10-5000 characters per comment |
| **Encoding** | UTF-8 recommended |

### Data Quality Tips

**✅ DO:**
- Remove duplicate comments
- Keep comments meaningful (not just "ok" or "...")
- Include variety (positive + negative)
- Use original customer language

**❌ AVOID:**
- Empty rows
- Only emojis
- URLs without context
- Spam/bot comments

### Common CSV Issues

**Issue 1: Wrong Delimiter**

If your CSV uses `;` or `\t` instead of `,`:

```bash
# Convert in Excel/Google Sheets
File → Save As → CSV (Comma delimited)
```

**Issue 2: Special Characters**

If you have quotes or commas in text:

```csv
review
"He said, ""Great product!"" and I agree."
```

**Issue 3: Encoding**

If you see weird characters (Ã©, â€™):

```bash
# Save with UTF-8 encoding
File → Save As → Encoding: UTF-8
```

---

## Using the Web Interface

### Step 1: Upload Your CSV

1. **Open TaskFlow** in your browser: `http://localhost:8000`
2. **Click** the upload area or **drag & drop** your CSV file
3. **Wait** for upload (1-5 seconds)

**What Happens:**
- System validates your file
- Auto-detects text column
- Shows preview of first 5 rows

**Screenshot:**
```
┌─────────────────────────────────────┐
│  Step 1: Upload CSV File            │
├─────────────────────────────────────┤
│  📁 Click to upload or drag & drop  │
│     Maximum file size: 10MB         │
└─────────────────────────────────────┘
```

**Success:**
```
✅ File: customer_reviews.csv
✅ Rows: 150
✅ Text Column: review
```

**If Error:**
- Check file is .csv format
- Ensure at least 10 rows
- Verify file size < 10MB

---

### Step 2: Configure Report

After successful upload, configure your report:

**A. Select Report Type**

| Type | When to Use | Length |
|------|-------------|--------|
| **Executive Summary** | Quick overview for management | 3-5 pages |
| **Marketing Analysis** | Campaign performance, audience insights | 5-7 pages |
| **Product Insights** | Feature feedback, improvement ideas | 7-10 pages |
| **Customer Service** | Support issues, common problems | 5-8 pages |
| **Comprehensive** | Full detailed analysis | 15-20 pages |

**B. Verify Text Column**

System auto-detects, but you can override:

```
Text Column: [review ▼]  ← Change if wrong
```

**C. Click "Generate Report"**

---

### Step 3: Wait for Processing

**What Happens:**
1. **Detecting column** (5%) - Finding text to analyze
2. **Validating data** (10%) - Cleaning & checking quality
3. **Analyzing comments** (20-65%) - Running 5 ML models
4. **Quality check** (65-70%) - Verifying results
5. **Generating report** (70-100%) - Creating professional report

**Progress Bar:**
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 45%

Analyzing 150 comments...
```

**Time Estimates:**

| Comments | Time |
|----------|------|
| 10-50 | 20-30 seconds |
| 50-200 | 30-60 seconds |
| 200-1000 | 60-120 seconds |
| 1000+ | 2-5 minutes |

**💡 Tip:** Keep browser tab open during processing!

---

### Step 4: Download Report

When complete, you'll see:

```
✅ Report Ready!

Quality Score: 87.5/100
Word Count: 1,247
Processing Time: 45.2s
Cost: $0.0154

[📄 Download Markdown] [📑 Download PDF] [🔄 New Report]
```

**Download Options:**

**Markdown (.md)**
- Plain text format
- Easy to edit
- GitHub compatible
- Lightweight

**PDF (.pdf)**
- Professional styling
- Ready to share
- Print-friendly
- Includes metadata

**💡 Tip:** Download both formats for flexibility!

---

## Understanding Your Report

### Report Structure

All reports follow this structure:

```
1. Executive Summary
   - Key findings at a glance
   - Main sentiment breakdown
   
2. Sentiment Analysis
   - Positive/Negative/Neutral distribution
   - Trends and patterns
   
3. Emotion Analysis
   - Joy, Anger, Sadness, Fear, Surprise
   - Emotional landscape
   
4. Key Topics
   - What customers talk about most
   - Sentiment per topic
   
5. Important Entities
   - Brands, products, features mentioned
   - How customers feel about each
   
6. Actionable Recommendations
   - What to do next
   - Priority actions
```

### Reading Sentiment

**Sentiment Distribution:**
```
Positive:  620 comments (62%)  🟢
Negative:  280 comments (28%)  🔴
Neutral:   100 comments (10%)  ⚪
```

**What This Means:**
- **62% positive:** Majority satisfied
- **28% negative:** Significant issues to address
- **10% neutral:** Undecided/mixed feelings

**Benchmarks:**

| Score | Interpretation |
|-------|----------------|
| 80%+ positive | Excellent |
| 60-80% positive | Good |
| 40-60% positive | Mixed |
| 20-40% positive | Concerning |
| <20% positive | Critical issues |

### Reading Emotions

**Emotion Distribution:**
```
Joy:      350 comments (35%)
Neutral:  400 comments (40%)
Anger:    150 comments (15%)
Sadness:   80 comments (8%)
Fear:      20 comments (2%)
```

**What This Means:**
- **High Joy:** Customers are delighted
- **High Neutral:** Room for memorable experiences
- **High Anger:** Urgent issues causing frustration
- **High Sadness:** Disappointment, unmet expectations
- **High Fear:** Concerns about safety/reliability

### Reading Topics

**Top Topics:**
```
1. Product Quality (320 mentions)
   Average Sentiment: +0.72 (Positive)
   
2. Customer Service (185 mentions)
   Average Sentiment: -0.45 (Negative)
   
3. Shipping (142 mentions)
   Average Sentiment: +0.23 (Slightly Positive)
```

**What This Means:**
- **Quality:** Customers love product quality (+0.72)
- **Service:** Service needs improvement (-0.45)
- **Shipping:** Mixed feelings, room to improve (+0.23)

### Understanding Recommendations

Reports include 3 types of recommendations:

**1. Immediate Actions**
- Critical issues requiring urgent attention
- Examples: "Address shipping delays", "Improve customer service response time"

**2. Short-term Improvements** (1-3 months)
- Important but not urgent
- Examples: "Enhance packaging", "Add product features"

**3. Long-term Strategy** (3-12 months)
- Strategic initiatives
- Examples: "Build loyalty program", "Expand product line"

---

## Report Types

### Executive Summary

**Best For:** C-level executives, board meetings, quick overview

**Contains:**
- High-level findings
- Key metrics
- Critical issues
- Top 3 recommendations

**Length:** 3-5 pages

**Example Use Case:**
> "I need to present customer sentiment to the board next week."

---

### Marketing Analysis

**Best For:** Marketing teams, campaign evaluation, audience insights

**Contains:**
- Sentiment trends
- Audience segmentation
- Campaign effectiveness
- Brand perception
- Competitor mentions

**Length:** 5-7 pages

**Example Use Case:**
> "We just launched a campaign and want to measure customer response."

---

### Product Insights

**Best For:** Product managers, development teams, roadmap planning

**Contains:**
- Feature feedback
- Pain points
- Improvement suggestions
- Feature requests
- Bug reports

**Length:** 7-10 pages

**Example Use Case:**
> "We need to prioritize features for our next release."

---

### Customer Service

**Best For:** Support teams, service managers, operations

**Contains:**
- Common issues
- Support effectiveness
- Resolution patterns
- Training needs
- Process improvements

**Length:** 5-8 pages

**Example Use Case:**
> "Our support tickets are increasing. What are the main issues?"

---

### Comprehensive

**Best For:** Detailed analysis, research, annual reviews

**Contains:**
- Everything from other reports
- Deep-dive analysis
- Statistical details
- Trend analysis
- Comparative insights

**Length:** 15-20 pages

**Example Use Case:**
> "We need a complete analysis for our annual customer feedback review."

---

## Best Practices

### Data Collection

**✅ DO:**
- Collect regularly (weekly/monthly)
- Mix sources (email, social, support)
- Keep original wording
- Note time periods
- Track changes over time

**❌ AVOID:**
- Only negative feedback
- Heavily filtered data
- Paraphrased comments
- Very old data (>1 year)

### Frequency

**Recommended Analysis Schedule:**

| Business Type | Frequency |
|---------------|-----------|
| E-commerce | Weekly |
| SaaS | Monthly |
| Retail | Bi-weekly |
| Services | Monthly |
| Campaigns | After each campaign |

### Sample Size

**Minimum:** 10 comments (system requirement)
**Recommended:** 50+ comments (reliable insights)
**Optimal:** 200+ comments (statistical significance)

**Rule of Thumb:**
- 10-50: Basic trends
- 50-200: Reliable insights
- 200-1000: High confidence
- 1000+: Statistical significance

### Combining Data

**Don't mix:**
- ❌ Different time periods (2023 + 2024)
- ❌ Different products
- ❌ Different languages

**Do combine:**
- ✅ Same product, multiple platforms
- ✅ Same time period, different channels
- ✅ Related product line

### Interpreting Results

**Quality Score Meaning:**

| Score | Quality | Action |
|-------|---------|--------|
| 90-100 | Excellent | Use with confidence |
| 80-89 | Good | Minor caveats in report |
| 70-79 | Fair | Review recommendations carefully |
| <70 | Low | Consider data quality issues |

**When Quality is Low:**
- Check if data is meaningful
- Ensure enough variety (not all "ok")
- Verify text is in English (if using default models)
- Try with more data

---

## FAQ

### General

**Q: How accurate is the sentiment analysis?**

A: 90%+ accuracy on English text. Using 5 specialized models for different aspects.

**Q: Can I analyze non-English text?**

A: Currently optimized for English. Other languages have lower accuracy. Contact for custom models.

**Q: Is my data private?**

A: Yes. Data is processed locally, deleted after analysis. Reports use anonymized insights only.

**Q: How much does it cost?**

A: ~$0.015 per 1000 comments. Significantly cheaper than manual analysis or paid tools ($200-500/month).

---

### Technical

**Q: What format should my CSV be?**

A: Standard CSV with at least one text column. Comma-separated, UTF-8 encoding recommended.

**Q: What if column detection is wrong?**

A: You can manually select the correct column in Step 2.

**Q: Can I cancel a job?**

A: Not currently. But you can start a new analysis anytime.

**Q: How long are reports stored?**

A: Reports are stored temporarily (24 hours). Download and save locally.

---

### Data

**Q: What's the maximum file size?**

A: 10MB file size, 10,000 rows maximum.

**Q: What if I have more than 10,000 comments?**

A: Split into multiple files or sample representative subset.

**Q: Can I upload the same file twice?**

A: Yes, but you'll get similar results. Use for different report types.

**Q: What about duplicate comments?**

A: System removes exact duplicates during validation.

---

### Reports

**Q: Can I edit the report?**

A: Download Markdown (.md) format and edit in any text editor.

**Q: Can I regenerate with different settings?**

A: Yes! Upload same file, select different report type.

**Q: Why is my report short?**

A: Limited data (few comments) generates shorter reports. Add more data for comprehensive analysis.

**Q: Can I customize report format?**

A: Not currently. Standard format ensures quality and consistency.

---

### Troubleshooting

**Q: Upload failed - "CSV format error"**

A: Check file has consistent columns. Open in Excel → Save As → CSV (Comma delimited).

**Q: "Column not found" error**

A: Selected column doesn't exist in your file. Use auto-detect or check column names.

**Q: Job failed - "Analysis quality too low"**

A: Data quality issues. Ensure comments are meaningful (not just "ok", "...", emojis).

**Q: Report quality score is low**

A: Check data quality:
- Remove very short comments
- Ensure variety (not all positive/negative)
- Use English text (or custom models)
- Add more data if possible

**Q: Processing is stuck**

A: Refresh page. If still stuck, restart server and try again.

---

## Support

**Need Help?**

📧 **Email:** support@taskflow.example.com  
💬 **GitHub Issues:** [github.com/yourusername/taskflow/issues](https://github.com/yourusername/taskflow/issues)  
📖 **Documentation:** [Full Docs](https://github.com/yourusername/taskflow/tree/main/docs)

---

## Tips for Better Results

### 1. Data Quality Matters

**Good data:**
```
"The product quality is excellent. Fast shipping and great customer service!"
"Disappointed with the purchase. Product broke after one week."
"It's okay, nothing special. Shipping was slow but acceptable."
```

**Poor data:**
```
"ok"
"..."
"👍👍👍"
```

### 2. Enough Context

**Good:**
```
"The new iPhone camera is amazing! Much better than my old Android."
```

**Too vague:**
```
"Good phone"
```

### 3. Variety

Include:
- ✅ Positive AND negative feedback
- ✅ Different aspects (product, service, shipping)
- ✅ Various lengths (short + detailed)

### 4. Regular Analysis

Don't wait! Analyze regularly:
- Catch issues early
- Track improvements
- Identify trends
- Measure changes

### 5. Act on Insights

Reports are only valuable if you act:
1. Read recommendations
2. Share with relevant teams
3. Prioritize actions
4. Implement changes
5. Measure impact (next analysis)

---

## Next Steps

**Ready to analyze?**

1. ✅ Prepare your CSV file
2. ✅ Go to `http://localhost:8000`
3. ✅ Upload and generate report
4. ✅ Download and share insights
5. ✅ Take action on recommendations!

**Want to learn more?**

- [API Documentation](API_DOCUMENTATION.md) - For developers
- [Developer Guide](DEVELOPER_GUIDE.md) - For contributors
- [Deployment Guide](DEPLOYMENT_GUIDE.md) - For production use

---

**Happy analyzing! 📊**