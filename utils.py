import torch
from transformers import BertTokenizer, BertForSequenceClassification
import html
import re
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet
import nltk
import os

# ─── NLTK Setup ────────────────────────────────────────────────────────────────
project_dir = os.path.dirname(os.path.abspath(__file__))
nltk_data_path = os.path.join(project_dir, "nltk_data")
nltk.data.path.append(nltk_data_path)

for pkg in ['punkt', 'averaged_perceptron_tagger', 'wordnet', 'omw-1.4',
            'averaged_perceptron_tagger_eng', 'punkt_tab']:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

lemmatizer = WordNetLemmatizer()

# ─── POS Mapper ────────────────────────────────────────────────────────────────
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'): return wordnet.ADJ
    if treebank_tag.startswith('V'): return wordnet.VERB
    if treebank_tag.startswith('R'): return wordnet.ADV
    return wordnet.NOUN

# ─── Text Cleaning ─────────────────────────────────────────────────────────────
def clean_and_lemmatize_text(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = html.unescape(text)
    try:
        text = text.encode('latin1', errors='ignore').decode('utf-8', errors='ignore')
    except Exception:
        pass
    text = re.sub(r"\bRT\b", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    lemmatized = [lemmatizer.lemmatize(t, get_wordnet_pos(p)) for t, p in pos_tags]
    return " ".join(lemmatized)

# ─── Text Stats ────────────────────────────────────────────────────────────────
def get_text_stats(text: str) -> dict:
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    word_count = len(words)
    sentence_count = max(len(sentences), 1)
    avg_words = round(word_count / sentence_count, 1)
    quality = "⚠️ Too short — add more context for better accuracy" if word_count < 8 \
        else "✅ Good length" if word_count < 50 \
        else "✅ Great detail"
    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_words_per_sentence": avg_words,
        "quality_hint": quality,
    }

# ─── Model ─────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(project_dir, "model")
DEVICE = "cpu"  # Streamlit Cloud has no GPU; CPU is the default

def load_model():
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(
        MODEL_PATH,
        dtype=torch.float32,
    )
    model.eval()
    return tokenizer, model

# ─── Labels ────────────────────────────────────────────────────────────────────
label_map = {
    0: "Anxiety",
    1: "Bipolar",
    2: "Depression",
    3: "Normal",
    4: "Personality Disorder",
    5: "Stress",
    6: "Suicidal",
}

label_colors = {
    "Anxiety":              "#FFD166",
    "Bipolar":              "#F4A261",
    "Depression":           "#E76F91",
    "Normal":               "#06D6A0",
    "Personality Disorder": "#9B72CF",
    "Stress":               "#FF9E6D",
    "Suicidal":             "#EF233C",
}

label_icons = {
    "Anxiety":              "😰",
    "Bipolar":              "🔄",
    "Depression":           "😔",
    "Normal":               "😊",
    "Personality Disorder": "🧩",
    "Stress":               "😤",
    "Suicidal":             "🆘",
}

# ─── Label Descriptions ────────────────────────────────────────────────────────
def get_label_description(label: str) -> str:
    descriptions = {
        "Anxiety": (
            "Anxiety is characterised by persistent worry, tension, and a sense of unease. "
            "The text shows signs of fear, nervousness, or excessive concern about everyday situations."
        ),
        "Bipolar": (
            "Bipolar disorder involves episodes of extreme mood swings — from manic highs to depressive lows. "
            "The text reflects shifting energy levels, grandiosity, or intense emotional states."
        ),
        "Depression": (
            "Depression is marked by persistent sadness, loss of interest, and low energy. "
            "The text expresses hopelessness, fatigue, or a lack of motivation."
        ),
        "Normal": (
            "The text reflects a balanced and healthy emotional state. "
            "No significant mental health indicators were detected."
        ),
        "Personality Disorder": (
            "Personality disorders affect how a person thinks, feels, and relates to others. "
            "The text may reflect unstable self-image, impulsivity, or turbulent relationships."
        ),
        "Stress": (
            "Stress is the body's reaction to challenging situations. "
            "The text shows signs of being overwhelmed, pressured, or burned out."
        ),
        "Suicidal": (
            "The text contains indicators associated with suicidal ideation — thoughts of self-harm or ending one's life. "
            "If this reflects your current state, please reach out for help immediately."
        ),
    }
    return descriptions.get(label, "")

# ─── Coping Strategies & Resources ────────────────────────────────────────────
def get_resources(label: str) -> dict:
    resources = {
        "Anxiety": {
            "is_crisis": False,
            "tips": [
                "🌬️ **Box breathing**: Inhale 4s → Hold 4s → Exhale 4s → Hold 4s",
                "🧘 Practice the **5-4-3-2-1 grounding technique** to stay present",
                "📵 Limit caffeine and social media before bed",
                "🚶 Take a short walk — physical movement reduces cortisol",
                "📓 Write down your worries to externalize and examine them",
            ],
            "links": [
                ("ADAA — Anxiety & Depression Association", "https://adaa.org"),
                ("MindTools — Stress & Anxiety Resources", "https://www.mindtools.com/"),
                ("Calm App", "https://www.calm.com"),
            ],
        },
        "Bipolar": {
            "is_crisis": False,
            "tips": [
                "🕐 Maintain a **consistent sleep schedule** — irregular sleep can trigger episodes",
                "📋 Keep a **mood log** to identify triggers and patterns",
                "🏃 Regular moderate exercise stabilises mood",
                "💊 Do not skip prescribed medications without consultation",
                "🤝 Join a bipolar support group to share experiences",
            ],
            "links": [
                ("DBSA — Depression & Bipolar Support Alliance", "https://www.dbsalliance.org"),
                ("NAMI Bipolar Disorder", "https://www.nami.org/About-Mental-Illness/Mental-Health-Conditions/Bipolar-Disorder"),
                ("Headspace", "https://www.headspace.com"),
            ],
        },
        "Depression": {
            "is_crisis": True,
            "tips": [
                "☀️ Get sunlight exposure within 1 hour of waking — it boosts serotonin",
                "📖 Journaling your feelings for 10 minutes daily can provide relief",
                "🤝 Reach out to one trusted person today — isolation worsens depression",
                "🏋️ Even 20 minutes of exercise has antidepressant effects",
                "🎯 Set one small, achievable goal for the day — don't overwhelm yourself",
            ],
            "links": [
                ("NAMI Depression Resources", "https://www.nami.org/About-Mental-Illness/Mental-Health-Conditions/Depression"),
                ("7 Cups — Free Emotional Support Chat", "https://www.7cups.com"),
                ("BetterHelp Online Therapy", "https://www.betterhelp.com"),
            ],
        },
        "Normal": {
            "is_crisis": False,
            "tips": [
                "🌿 Keep nurturing your mental wellness with regular self-check-ins",
                "😴 Prioritise 7-9 hours of quality sleep each night",
                "🧠 Practice mindfulness or meditation for 5-10 minutes daily",
                "👥 Maintain meaningful social connections",
                "🎨 Engage in a creative hobby that brings you joy",
            ],
            "links": [
                ("WHO Mental Health", "https://www.who.int/health-topics/mental-health"),
                ("Headspace", "https://www.headspace.com"),
                ("Mental Health America", "https://www.mhanational.org"),
            ],
        },
        "Personality Disorder": {
            "is_crisis": False,
            "tips": [
                "📚 Look into **DBT (Dialectical Behaviour Therapy)** — highly effective for BPD",
                "🧘 Practise mindfulness to observe emotions without reacting impulsively",
                "📓 Keep a diary card to track emotions and urges",
                "🤝 Seek support from a trained mental health professional",
                "🌱 Recovery is possible — with consistency and professional support",
            ],
            "links": [
                ("TARA for BPD", "https://www.tara4bpd.org"),
                ("DBT Self-Help", "https://www.dbtselfhelp.com"),
                ("Psychology Today — Find a Therapist", "https://www.psychologytoday.com/us/therapists"),
            ],
        },
        "Stress": {
            "is_crisis": False,
            "tips": [
                "📝 Make a **priority list** — tackle the most critical task first",
                "⏱️ Use the **Pomodoro Technique** — 25 min focus, 5 min break",
                "🚫 Learn to say **no** — protect your time and energy",
                "💧 Stay hydrated and eat balanced meals — stress depletes nutrients",
                "🎵 Listen to music or nature sounds to activate your parasympathetic system",
            ],
            "links": [
                ("APA Stress Management", "https://www.apa.org/topics/stress"),
                ("Insight Timer — Free Meditation", "https://insighttimer.com"),
                ("Stress Management Society", "https://www.stress.org.uk"),
            ],
        },
        "Suicidal": {
            "is_crisis": True,
            "tips": [
                "📞 **Please call or text 988** (Suicide & Crisis Lifeline) right now",
                "🏥 Go to your nearest emergency room if you are in immediate danger",
                "🤝 Tell someone you trust how you are feeling — don't face this alone",
                "🔒 Remove access to means of self-harm if possible",
                "💬 A trained counsellor is available 24/7 — your life has value",
            ],
            "links": [
                ("988 Suicide & Crisis Lifeline (Call/Text 988)", "https://988lifeline.org"),
                ("Crisis Text Line — Text HOME to 741741", "https://www.crisistextline.org"),
                ("International Association for Suicide Prevention", "https://www.iasp.info/resources/Crisis_Centres/"),
            ],
        },
    }
    return resources.get(label, {"is_crisis": False, "tips": [], "links": []})

# ─── Global Crisis Info ────────────────────────────────────────────────────────
CRISIS_INFO = {
    "title": "🆘 Crisis Support",
    "lines": [
        ("📞 988 Suicide & Crisis Lifeline (US)", "tel:988", "Call or text 988"),
        ("💬 Crisis Text Line", "https://www.crisistextline.org", "Text HOME to 741741"),
        ("🌍 International Crisis Centres", "https://www.iasp.info/resources/Crisis_Centres/", "Find support near you"),
    ],
}
