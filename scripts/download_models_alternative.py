# scripts/download_models_alternative.py
from transformers import pipeline
from sentence_transformers import SentenceTransformer

print("Downloading alternative models (smaller, safer)...")

# 1. Sentiment - Use DistilBERT (smaller, has safetensors)
print("\n1/5 Downloading Sentiment model...")
pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# 2. Emotion - Keep same
print("\n2/5 Downloading Emotion model...")
pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# 3. Topic - Keep same
print("\n3/5 Downloading Topic model...")
SentenceTransformer('all-MiniLM-L6-v2')

# 4. Entity - Keep same
print("\n4/5 Downloading Entity model...")
pipeline("ner", model="dslim/bert-base-NER")

# 5. Keyphrase - Use simpler model
print("\n5/5 Downloading Keyphrase model...")
pipeline("feature-extraction", model="distilbert-base-uncased")

print("\n✅ All models downloaded!")