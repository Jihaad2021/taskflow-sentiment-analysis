# TaskFlow Performance Optimization Guide

> **Practical tips untuk improve speed, reduce cost, dan scale the system**

---

## Current Performance Baseline

### With Mock LLM (Testing)
- 60 comments: ~87s (Analysis: 60s, Report: 0.5s)
- Cost: $0.003 (mock)

### With Real LLM (Production)
- 60 comments: ~105s (Analysis: 60s, Report: 20s)
- Cost: $0.015 (Claude 3.5 Sonnet)

**Bottleneck:** Analysis layer (5 ML models sequential processing)

---

## Optimization Strategies

### 1. GPU Acceleration

**Current:** CPU-only inference  
**Target:** GPU (CUDA) inference

**Implementation:**
```python
# Change device from 'cpu' to 'cuda'
orchestrator = AnalysisOrchestratorAgent(device="cuda")
```

**Expected Improvement:**
- Speed: 10x faster (60s → 6s for analysis)
- Cost: Same (self-hosted)
- Trade-off: Need GPU ($0.50-2/hour on cloud)

**When to use:**
- Processing >100 comments regularly
- Real-time analysis needed
- Budget allows GPU costs

---

### 2. Batch Processing

**Current:** Process comments one-by-one  
**Target:** Process in batches of 32-128

**Implementation:**
```python
# In src/agents/orchestrator.py
def _analyze_batch(self, comments: List[str], batch_size: int = 32):
    """Process comments in batches for speed."""
    
    all_results = []
    for i in range(0, len(comments), batch_size):
        batch = comments[i:i+batch_size]
        
        # Process batch with each tool
        sentiment_batch = self.sentiment_tool.analyze_batch(batch)
        emotion_batch = self.emotion_tool.analyze_batch(batch)
        # ... other tools
        
        all_results.extend(zip(sentiment_batch, emotion_batch, ...))
    
    return all_results
```

**Expected Improvement:**
- Speed: 2-3x faster
- Cost: Same
- Trade-off: Higher memory usage

**Best for:**
- Large datasets (>500 comments)
- GPU available
- Memory not constrained

---

### 3. Async/Parallel Tool Execution

**Current:** Run tools sequentially  
**Target:** Run all 5 tools in parallel

**Implementation:**
```python
import asyncio

async def _analyze_parallel(self, comment: str):
    """Run all tools in parallel."""
    
    tasks = [
        asyncio.to_thread(self.sentiment_tool.analyze, comment),
        asyncio.to_thread(self.emotion_tool.analyze, comment),
        asyncio.to_thread(self.topic_tool.analyze, comment),
        asyncio.to_thread(self.entity_tool.analyze, comment),
        asyncio.to_thread(self.keyphrase_tool.analyze, comment),
    ]
    
    results = await asyncio.gather(*tasks)
    return results
```

**Expected Improvement:**
- Speed: 3-4x faster (if CPU has 5+ cores)
- Cost: Same
- Trade-off: Higher CPU usage, complex code

**Best for:**
- Multi-core CPUs available
- Moderate dataset sizes
- Want speed without GPU

---

### 4. Model Quantization

**Current:** FP32 (full precision)  
**Target:** INT8 or FP16 (quantized)

**Implementation:**
```python
from transformers import AutoModelForSequenceClassification
from optimum.onnxruntime import ORTModelForSequenceClassification

# Convert to ONNX + quantize
model = ORTModelForSequenceClassification.from_pretrained(
    model_name,
    export=True,
    provider="CPUExecutionProvider"
)
```

**Expected Improvement:**
- Speed: 2-3x faster
- Memory: 4x less
- Accuracy: 1-2% drop (acceptable)
- Cost: Same

**Trade-off:**
- Setup complexity
- Slight accuracy loss

**Best for:**
- CPU-only deployment
- Memory constrained
- Speed critical

---

### 5. Caching Strategy

**What to cache:**
- Model outputs for identical text
- LLM responses for same prompts
- Analysis results for duplicate comments

**Implementation:**
```python
from functools import lru_cache
import hashlib

class CachedOrchestrator(AnalysisOrchestratorAgent):
    
    @lru_cache(maxsize=1000)
    def _analyze_comment_cached(self, text_hash: str, text: str):
        """Cache results by text hash."""
        return super()._analyze_comment(text)
    
    def _analyze_comment(self, text: str):
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return self._analyze_comment_cached(text_hash, text)
```

**Expected Improvement:**
- Speed: 10-100x for cached items
- Cost: Same
- Memory: +100MB for 1000 cached items

**Best for:**
- Many duplicate comments
- Repeated analyses
- A/B testing

---

### 6. Reduce LLM Token Usage

**Current:** Full context in every prompt  
**Target:** Summarize data, optimize prompts

**Implementation:**
```python
# Before: Send all 100 topics
prompt = f"Topics: {all_topics}"

# After: Send top 10 only
prompt = f"Top Topics: {top_topics[:10]}"
```

**Optimizations:**
- Top N items only (not all data)
- Shorter prompts (remove fluff)
- Lower temperature (0.3 vs 0.7) for consistent output
- Smaller max_tokens where possible

**Expected Improvement:**
- Speed: 20-30% faster LLM calls
- Cost: 30-40% cheaper
- Quality: Minimal impact

---

### 7. Model Selection

**Current Models:**

| Tool | Model | Size | Speed |
|------|-------|------|-------|
| Sentiment | twitter-roberta-base | 125M | Fast |
| Emotion | emotion-distilroberta | 82M | Fast |
| Topic | all-MiniLM-L6-v2 | 22M | Very Fast |
| Entity | bert-base-NER | 110M | Medium |
| Keyphrase | keyphrase-distilbert | 66M | Fast |

**Faster Alternatives:**

| Tool | Alt Model | Size | Speedup |
|------|-----------|------|---------|
| Sentiment | distilbert-base-sentiment | 66M | 2x |
| Entity | spacy-en-core-sm | 12M | 5x |
| Topic | Use keywords only | - | 10x |

**Trade-off:** 3-5% accuracy drop, 2-5x speed increase

---

### 8. Skip Optional Tools

**For simple reports, can skip:**
- KeyphraseTool (use top topics instead)
- EntityTool (if no brand mentions needed)
- TopicTool (if sentiment only needed)

**Implementation:**
```python
class FastOrchestrator(AnalysisOrchestratorAgent):
    """Minimal orchestrator for speed."""
    
    def __init__(self, tools: List[str] = ["sentiment", "emotion"]):
        self.enabled_tools = tools
        # Only load enabled tools
```

**Expected Improvement:**
- Speed: 40-60% faster (2 tools vs 5)
- Cost: Same
- Quality: Reduced (less data)

**Best for:**
- Simple dashboards
- Real-time processing
- High-volume scenarios

---

### 9. Database Optimization

**For production with history:**
```python
# Use indexes
CREATE INDEX idx_comment_hash ON comments(hash);
CREATE INDEX idx_created_at ON reports(created_at);

# Cache frequent queries
@cached(ttl=3600)
def get_recent_reports():
    return db.query(Report).order_by(desc(created_at)).limit(10)
```

**Expected Improvement:**
- Query speed: 10-100x faster
- API latency: 50-200ms reduction

---

### 10. API Response Optimization

**Reduce response size:**
```python
# Before: Return full analysis
{
    "individual_results": [...1000 items...]  # 5MB
}

# After: Return summary only
{
    "summary": {...},  # 50KB
    "download_url": "/api/full-results/123"  # Link to full data
}
```

**Expected Improvement:**
- Response time: 50-80% faster
- Bandwidth: 90% reduction

---

## Optimization Combinations

### Combo 1: Cost Optimization (Target: <$0.01/report)
```
✓ Reduce LLM tokens (30% savings)
✓ Cache duplicate prompts
✓ Lower temperature (0.3)
✓ Smaller max_tokens

Result: $0.015 → $0.008 (47% reduction)
```

### Combo 2: Speed Optimization (Target: <30s total)
```
✓ GPU acceleration (10x)
✓ Parallel tool execution (3x)
✓ Batch processing (2x)

Result: 87s → 25s (65x cumulative speedup)
```

### Combo 3: Memory Optimization (Target: <2GB)
```
✓ Model quantization (4x)
✓ Batch size: 16 (vs 32)
✓ Clear cache after processing

Result: 4GB → 1.5GB
```

### Combo 4: Scale Optimization (Target: 10K comments)
```
✓ GPU + Batch + Async
✓ Skip optional tools
✓ LRU cache (1000 items)

Result: 10K comments in ~5 minutes (vs 2 hours)
```

---

## Implementation Priority

### Phase 1: Quick Wins (1-2 days)
1. ✅ Reduce LLM token usage
2. ✅ Optimize prompts
3. ✅ Lower LLM temperature

**Impact:** 30% cost reduction, 20% speed increase

### Phase 2: Medium Effort (3-5 days)
1. Batch processing
2. Simple caching
3. Skip optional tools (configurable)

**Impact:** 2x speed, same cost

### Phase 3: High Effort (1-2 weeks)
1. GPU support
2. Async/parallel execution
3. Model quantization

**Impact:** 10x speed, same cost

### Phase 4: Advanced (2-4 weeks)
1. Distributed processing (Celery)
2. Custom fine-tuned models
3. Real-time streaming

**Impact:** Scale to millions of comments

---

## Monitoring & Profiling

### Add Performance Tracking
```python
import time
from functools import wraps

def profile(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        
        # Log to monitoring
        logger.info(f"{func.__name__}: {duration:.2f}s")
        
        return result
    return wrapper

# Use
@profile
def analyze_comments(self, comments):
    ...
```

### Key Metrics to Track

1. **Latency:**
   - P50, P95, P99 per agent
   - Total pipeline time
   - Per-tool inference time

2. **Throughput:**
   - Comments/second
   - Reports/hour
   - Tokens/minute (LLM)

3. **Resource Usage:**
   - CPU utilization
   - Memory consumption
   - GPU utilization (if available)

4. **Cost:**
   - LLM tokens used
   - Cost per report
   - Cost per comment

5. **Quality:**
   - Report scores over time
   - Regeneration rates
   - Error rates

---

## Cloud Deployment Options

### Option 1: CPU-Only (Budget)

**Hardware:** 4 vCPU, 8GB RAM  
**Cost:** $40-80/month  
**Performance:** 60 comments in 90s  
**Best for:** <1000 reports/month

### Option 2: GPU (Performance)

**Hardware:** 1x T4 GPU, 4 vCPU, 16GB RAM  
**Cost:** $300-500/month  
**Performance:** 60 comments in 15s  
**Best for:** >5000 reports/month

### Option 3: Serverless (Scale)

**Hardware:** AWS Lambda (CPU)  
**Cost:** $0.20/1M seconds  
**Performance:** Auto-scales  
**Best for:** Unpredictable traffic

### Option 4: Hybrid

**Hardware:** CPU for API, GPU for batch jobs  
**Cost:** $100-200/month  
**Performance:** Best of both  
**Best for:** Mixed workload

---

## Benchmarking Scripts

Add to `scripts/benchmark.py`:
```python
"""Benchmark script for performance testing."""

import time
import pandas as pd
from src.agents.orchestrator import AnalysisOrchestratorAgent

def benchmark_analysis(num_comments: int, device: str = "cpu"):
    """Benchmark analysis performance."""
    
    # Generate test data
    comments = [f"Test comment {i}" for i in range(num_comments)]
    
    # Run analysis
    orchestrator = AnalysisOrchestratorAgent(device=device)
    
    start = time.time()
    result = orchestrator.execute(
        AnalysisOrchestratorInput(comments=comments)
    )
    duration = time.time() - start
    
    # Report
    print(f"Comments: {num_comments}")
    print(f"Device: {device}")
    print(f"Total time: {duration:.2f}s")
    print(f"Comments/sec: {num_comments/duration:.1f}")
    print(f"Avg per comment: {duration/num_comments*1000:.1f}ms")

if __name__ == "__main__":
    for size in [10, 50, 100, 500]:
        benchmark_analysis(size, "cpu")
```

---

## Summary Checklist

### Before Optimization
- [ ] Profile current performance
- [ ] Identify bottlenecks
- [ ] Set target metrics
- [ ] Measure baseline

### Quick Wins (Do First)
- [ ] Optimize LLM prompts
- [ ] Reduce token usage
- [ ] Lower temperature
- [ ] Cache duplicate calls

### Medium Effort
- [ ] Batch processing
- [ ] Simple caching
- [ ] Skip optional tools

### High Effort (If Needed)
- [ ] GPU acceleration
- [ ] Model quantization
- [ ] Async processing

### Monitoring
- [ ] Add performance logging
- [ ] Track key metrics
- [ ] Set up alerts
- [ ] Regular profiling

---

**Last Updated:** December 2024  
**Status:** Week 3 Complete, Optimization Guide Ready