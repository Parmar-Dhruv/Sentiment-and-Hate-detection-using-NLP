"""
E5 — Gradio Demo
Social Media Sentiment and Hate Detection

Loads both fine-tuned checkpoints (E2 sentiment model, hate detection model)
once at module level and serves them via a tabbed Gradio interface.

Run locally with:
    python app.py
"""

import re

import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SENTI_MODEL_PATH = r"C:\Coding\Projects\Social_Media_Sentiment_Analysis_and_Hate_Dectection_System\others\sentiment_model"
HATE_MODEL_PATH = r"C:\Coding\Projects\Social_Media_Sentiment_Analysis_and_Hate_Dectection_System\others\hate_model"

MAX_LENGTH = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Preprocessing — kept byte-for-byte identical to the function used during
# E2 training-data preparation. Do NOT "fix" the @user lookahead edge case;
# the model was trained on this exact behavior, so matching it exactly is
# what preserves train/inference consistency, bug included.
# ---------------------------------------------------------------------------


def clean_text(text):

    # 1. Lowercase
    lower_text = text.lower()

    # 2. User anonymization
    mention_pattern = r'@(?!user)\w+'
    anony_text = re.sub(mention_pattern, ' @user', lower_text)

    # 3. URL Removal
    no_url_text = re.sub(r'https?://\S+', '', anony_text)

    # 4. Stripping hashtag but keeping hastag content
    removed_hash_text = no_url_text.replace("#", "")

    # 5. Normalize whitespace(strip + collapse internal spaces)
    stripped_text = removed_hash_text.strip()
    cleaned_text = " ".join(stripped_text.split())

    return cleaned_text


# ---------------------------------------------------------------------------
# Model loading — once at import time, reused across every request.
# Label mapping is pulled from each checkpoint's own config.json rather than
# hardcoded, since these are fine-tuned checkpoints and the config should
# already carry the correct id2label. If it doesn't, that's a real bug and
# it should surface in the demo output, not be papered over here.
# ---------------------------------------------------------------------------

print(f"Loading models on device: {DEVICE}")

senti_tokenizer = AutoTokenizer.from_pretrained(SENTI_MODEL_PATH)
senti_model = AutoModelForSequenceClassification.from_pretrained(SENTI_MODEL_PATH)
senti_model.to(DEVICE)
senti_model.eval()

hate_tokenizer = AutoTokenizer.from_pretrained(HATE_MODEL_PATH)
hate_model = AutoModelForSequenceClassification.from_pretrained(HATE_MODEL_PATH)
hate_model.to(DEVICE)
hate_model.eval()

print("Sentiment model id2label:", senti_model.config.id2label)
print("Hate model id2label:", hate_model.config.id2label)


# ---------------------------------------------------------------------------
# Inference functions
# ---------------------------------------------------------------------------


def _predict(text, tokenizer, model):
    """Shared inference path: clean -> tokenize -> forward -> softmax dict."""
    if not text or not text.strip():
        return {}

    processed = clean_text(text)

    inputs = tokenizer(
        processed,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()

    id2label = model.config.id2label
    return {id2label[i]: float(probs[i]) for i in range(len(probs))}


def predict_sentiment(text):
    return _predict(text, senti_tokenizer, senti_model)


def predict_hate(text):
    return _predict(text, hate_tokenizer, hate_model)


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

with gr.Blocks(title="Social Media Sentiment & Hate Detection") as demo:
    gr.Markdown("# Social Media Sentiment and Hate Detection")
    gr.Markdown(
        "Fine-tuned RoBERTa models on TweetEval. Input is preprocessed with "
        "the same `clean_text()` used at training time before inference."
    )

    with gr.Tab("Sentiment Analysis"):
        senti_input = gr.Textbox(
            label="Tweet text",
            placeholder="Type or paste a tweet...",
            lines=3,
        )
        senti_button = gr.Button("Analyze Sentiment", variant="primary")
        senti_output = gr.Label(label="Sentiment", num_top_classes=3)

        senti_button.click(fn=predict_sentiment, inputs=senti_input, outputs=senti_output)
        senti_input.submit(fn=predict_sentiment, inputs=senti_input, outputs=senti_output)

    with gr.Tab("Hate Detection"):
        hate_input = gr.Textbox(
            label="Tweet text",
            placeholder="Type or paste a tweet...",
            lines=3,
        )
        hate_button = gr.Button("Analyze for Hate Speech", variant="primary")
        hate_output = gr.Label(label="Hate Detection", num_top_classes=2)

        hate_button.click(fn=predict_hate, inputs=hate_input, outputs=hate_output)
        hate_input.submit(fn=predict_hate, inputs=hate_input, outputs=hate_output)


if __name__ == "__main__":
    demo.launch()
