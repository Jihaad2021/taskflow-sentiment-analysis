# scripts/download_models_safe.py
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import os

# Set environment variable to prefer safetensors
os.environ['TRANSFORMERS_SAFE_MODE'] = '1'

print("Downloading models with safetensors format...")

# 1. Sentiment
print("\n1/5 Downloading Sentiment model...")
try:
    pipeline(
        "sentiment-analysis", 
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        model_kwargs={"use_safetensors": True}
    )
    print("✅ Sentiment downloaded")
except Exception as e:
    print(f"⚠️ Error: {e}")

# 2. Emotion
print("\n2/5 Downloading Emotion model...")
try:
    pipeline(
        "text-classification", 
        model="j-hartmann/emotion-english-distilroberta-base",
        model_kwargs={"use_safetensors": True}
    )
    print("✅ Emotion downloaded")
except Exception as e:
    print(f"⚠️ Error: {e}")

# 3. Topic
print("\n3/5 Downloading Topic model...")
try:
    SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Topic downloaded")
except Exception as e:
    print(f"⚠️ Error: {e}")

# 4. Entity
print("\n4/5 Downloading Entity model...")
try:
    pipeline(
        "ner", 
        model="dslim/bert-base-NER",
        model_kwargs={"use_safetensors": True}
    )
    print("✅ Entity downloaded")
except Exception as e:
    print(f"⚠️ Error: {e}")

# 5. Keyphrase
print("\n5/5 Downloading Keyphrase model...")
try:
    pipeline(
        "text2text-generation", 
        model="ml6team/keyphrase-extraction-distilbert-inspec",
        model_kwargs={"use_safetensors": True}
    )
    print("✅ Keyphrase downloaded")
except Exception as e:
    print(f"⚠️ Error: {e}")

print("\n🎉 All models downloaded!")