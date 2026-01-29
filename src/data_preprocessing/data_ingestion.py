import glob
import os
import json
import re
import hashlib
from collections import Counter
import nltk
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from openai import OpenAI
from src.env import env  # Assuming this loads your .env file

# ──── CONFIG ────────────────────────────────────────────────────────────────
INPUT_DIR = "/home/sparkout/projects/llm fine tunning/data/sample data"
OUTPUT_DIR = "/home/sparkout/projects/llm fine tunning/data/cleaned data"
OUTPUT_JSONL = os.path.join(OUTPUT_DIR, "sparkout-qa-pairs.jsonl")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# DeepSeek API setup
DEEPSEEK_API_KEY = env.DEEPSEEK_API_KEY  # from your .env
DEEPSEEK_BASE_URL = env.MODEL_BASE_URL
MODEL_NAME = env.MODEL_NAME  # or "deepseek-chat" if you want non-reasoner

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)

MIN_CHUNK_WORDS = 15

# ──── Helpers ──────────────────────────────────────────────────────────────
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'http\S+|www\S+|@\S+|#\S+', '', text)
    return text

def remove_stopwords(text):
    words = word_tokenize(text.lower())
    filtered = [w for w in words if w not in stop_words and w.isalpha()]
    return ' '.join(filtered)

# ──── Load & clean files ───────────────────────────────────────────────────
print("Loading and cleaning files...")

all_unique_chunks = []
seen = set()

for fp in glob.glob(os.path.join(INPUT_DIR, "*.txt")):
    try:
        with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
            raw = f.read()
    except Exception as e:
        print(f"Skip {fp}: {e}")
        continue

    cleaned = clean_text(raw)
    if len(cleaned.split()) < MIN_CHUNK_WORDS * 2:
        continue

    # Split into chunks
    sentences = sent_tokenize(cleaned)
    current = ""
    chunks = []
    for s in sentences:
        if len(current) + len(s) > 1500 and current:
            chunks.append(current.strip())
            current = s
        else:
            current += " " + s if current else s
    if current:
        chunks.append(current.strip())

    # Filter & exact dedup
    for chunk in chunks:
        if len(chunk.split()) < MIN_CHUNK_WORDS:
            continue
        h = hashlib.sha256(chunk.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            all_unique_chunks.append(chunk)

print(f"Found {len(all_unique_chunks)} unique meaningful chunks")

# # ──── Generate Q&A with DeepSeek Reasoner ──────────────────────────────────
# qa_pairs = []

# print(f"Generating questions with {MODEL_NAME} via DeepSeek API...")

# for i, chunk in enumerate(all_unique_chunks, 1):
#     prompt = f"""You are creating high-quality training data for a company chatbot.

# Given this cleaned text from Sparkout Tech website:

# {chunk[:1400]}

# Generate **1–2 very natural, realistic questions** a real user might ask about this exact content.
# Return **only** valid JSON array — nothing else before or after:

# [
#   {{"question": "question here", "answer": "{chunk.replace('"', '\\"')}"}}
# ]

# Examples of good questions:
# - What AI chatbot services does Sparkout Tech provide?
# - How does Sparkout help with blockchain projects?
# - Where are Sparkout Tech offices located?
# """

#     try:
#         response = client.chat.completions.create(
#             model=MODEL_NAME,
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.7,
#             max_tokens=400,
#             response_format={"type": "json_object"}
#         )

#         raw = response.choices[0].message.content.strip()

#         # Extract and parse JSON
#         try:
#             data = json.loads(raw)
#             # DeepSeek often returns {"choices": [...]}, so adjust
#             items = data.get("choices", data) if isinstance(data, dict) else data
#             if isinstance(items, list):
#                 for item in items:
#                     if isinstance(item, dict) and "question" in item and "answer" in item:
#                         qa_pairs.append(item)
#                 print(f"Chunk {i}/{len(all_unique_chunks)}: OK")
#             else:
#                 print(f"Chunk {i}: Invalid format")
#         except json.JSONDecodeError:
#             print(f"Chunk {i}: JSON parse failed")

#     except Exception as e:
#         print(f"Chunk {i} API error: {e}")

# # ──── Save final JSONL ─────────────────────────────────────────────────────
# with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
#     for pair in qa_pairs:
#         f.write(json.dumps(pair, ensure_ascii=False) + '\n')

# print(f"\nFinished!")
# print(f"Saved {len(qa_pairs)} Q&A pairs to: {OUTPUT_JSONL}")

# if qa_pairs:
#     print("\nFirst 2 examples:")
#     print(json.dumps(qa_pairs[:2], indent=2, ensure_ascii=False))
# else:
#     print("No QA pairs generated. Check API key, model name, or chunk content.")



# ──── Gemini API Setup (replace DeepSeek block) ────────────────────────────
import google.generativeai as genai
from src.env import env

GEMINI_API_KEY = env.GEMINI_API_KEY
genai.configure(api_key=GEMINI_API_KEY)

# Use Gemini 1.5 Flash (free tier, very fast & good)
MODEL_NAME = env.GEMINI_MODEL_NAME # or "gemini-2.0-flash" if available

print(f"Using Gemini model: {MODEL_NAME}")

# ──── Generate Q&A with Gemini ─────────────────────────────────────────────
qa_pairs = []

print(f"Generating questions with {MODEL_NAME}...")

for i, chunk in enumerate(all_unique_chunks, 1):
    prompt = f"""You are creating high-quality training data for a company chatbot.

Given this cleaned text from Sparkout Tech website:

{chunk[:1400]}

Generate **1–2 very natural, realistic questions** a real user might ask about this exact content.
Return **only** valid JSON array — nothing else before or after:

[
  {{"question": "question here", "answer": "{chunk.replace('"', '\\"')}"}}
]

Examples of good questions:
- What AI chatbot services does Sparkout Tech provide?
- How does Sparkout help with blockchain projects?
- Where are Sparkout Tech offices located?
"""

    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=400,
                response_mime_type="application/json"
            )
        )

        raw = response.text.strip()

        # Extract JSON
        try:
            # Gemini sometimes wraps in markdown code block → clean it
            raw = re.sub(r'^```json\n|\n```$', '', raw.strip())
            items = json.loads(raw)
            for item in items:
                qa_pairs.append(item)
            print(f"Chunk {i}/{len(all_unique_chunks)}: OK")
        except json.JSONDecodeError:
            print(f"Chunk {i}: JSON parse failed")

    except Exception as e:
        print(f"Chunk {i} Gemini API error: {e}")

# ──── Save remains the same ────────────────────────────────────────────────
with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
    for pair in qa_pairs:
        f.write(json.dumps(pair, ensure_ascii=False) + '\n')

print(f"\nFinished!")
print(f"Saved {len(qa_pairs)} Q&A pairs to: {OUTPUT_JSONL}")