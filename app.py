import csv
import io

import streamlit as st
from transformers import pipeline

MODEL_ID = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
SAMPLE_REVIEWS = [
    ("I absolutely love this product! It works perfectly.", "POSITIVE"),
    ("The product arrived broken. Very disappointing.", "NEGATIVE"),
    ("Excellent quality and fast delivery. Highly recommended!", "POSITIVE"),
    ("Terrible customer service. I want a refund.", "NEGATIVE"),
    ("Delivery was slow, but the product is really good.", "POSITIVE"),
]

st.set_page_config(page_title="ReviewSense AI | Customer Sentiment", page_icon="💬", layout="wide")
st.markdown("""
<style>
.stApp {background: linear-gradient(125deg, #081226, #122445 60%, #10172e); color: #f8fafc;}
[data-testid="stSidebar"] {background: #101e37;}
[data-testid="stSidebar"] * {color: #edf2ff;}
.hero {background: linear-gradient(100deg, #213f86, #473499); padding: 1.6rem 2rem;
  border-radius: 18px; border: 1px solid #6a72bb; margin-bottom: 1.2rem;}
.hero h1 {color: white; margin: 0; font-size: 2.2rem;}
.hero p {color: #e0e7ff; margin: .5rem 0 0;}
[data-testid="stMetric"] {background:#1b2b4b; border:1px solid #455a80; border-radius:14px; padding:15px;}
[data-testid="stMetric"] * {color:#f8fafc !important; opacity:1 !important;}
.stButton > button[kind="primary"] {background:#6554df; color:white; border:none;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Downloading/loading DistilBERT sentiment model (first run may take a while)…")
def load_classifier():
    return pipeline(task="sentiment-analysis", model=MODEL_ID, device=-1)


def analyse(reviews, expected=None):
    predictions = load_classifier()(reviews, truncation=True, batch_size=8)
    rows = []
    for idx, (review, result) in enumerate(zip(reviews, predictions), start=1):
        label = result["label"].upper()
        row = {"#": idx, "Customer review": review, "Prediction": label,
               "Confidence (%)": round(100 * float(result["score"]), 2)}
        if expected is not None:
            target = expected[idx - 1]
            row["Expected"] = target
            row["Match"] = "Yes" if label == target else "No"
        rows.append(row)
    return rows


def csv_bytes(rows):
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")

st.markdown('<div class="hero"><h1>💬 ReviewSense AI</h1><p>Customer Review Sentiment Analysis · Pre-trained DistilBERT Transformer</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Project details")
    st.write("**Model:** DistilBERT (SST-2)")
    st.write("**Output:** POSITIVE / NEGATIVE")
    st.write("**API key:** Not required")
    st.caption("Predictions are model estimates. Confidence is not a guarantee of correctness.")
    st.divider()
    mode = st.radio("Choose a task", ["Test five sample reviews", "Analyse my own review", "Analyse multiple reviews"])

if mode == "Test five sample reviews":
    st.subheader("🧪 Assignment: Five customer reviews")
    st.write("Run the pre-trained Transformer on five labelled customer reviews and compare predicted sentiment with expected sentiment.")
    if st.button("Analyse five reviews", type="primary", use_container_width=True):
        try:
            reviews, targets = zip(*SAMPLE_REVIEWS)
            st.session_state["sample_rows"] = analyse(list(reviews), list(targets))
        except Exception as exc:
            st.error(f"Model could not load or predict: {exc}")
    if "sample_rows" in st.session_state:
        rows = st.session_state["sample_rows"]
        matches = sum(row["Match"] == "Yes" for row in rows)
        col1, col2, col3 = st.columns(3)
        col1.metric("Reviews analysed", len(rows))
        col2.metric("Matching labels", f"{matches}/{len(rows)}")
        col3.metric("Agreement on these 5 examples", f"{100 * matches / len(rows):.0f}%")
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download CSV results", csv_bytes(rows), file_name="sentiment_results.csv", mime="text/csv")
        st.info("Agreement on five hand-labelled examples is not the model's general test accuracy.")
elif mode == "Analyse my own review":
    st.subheader("✍️ Single review sentiment")
    review = st.text_area("Enter a customer review", "The delivery was late, but the product is excellent.", height=150)
    if st.button("Predict sentiment", type="primary"):
        if not review.strip():
            st.warning("Enter a review first.")
        else:
            try:
                row = analyse([review.strip()])[0]
                col1, col2 = st.columns(2)
                col1.metric("Predicted sentiment", row["Prediction"])
                col2.metric("Model confidence", f'{row["Confidence (%)"]:.2f}%')
            except Exception as exc:
                st.error(f"Model could not load or predict: {exc}")
else:
    st.subheader("📋 Batch sentiment classification")
    raw = st.text_area("Enter one review per line", "Excellent product, highly recommended!\nPoor packaging and slow delivery.", height=180)
    if st.button("Analyse all reviews", type="primary"):
        reviews = [line.strip() for line in raw.splitlines() if line.strip()]
        if not reviews:
            st.warning("Enter at least one non-empty review.")
        elif len(reviews) > 30:
            st.warning("For this demo, enter up to 30 reviews at a time.")
        else:
            try:
                st.session_state["batch_rows"] = analyse(reviews)
            except Exception as exc:
                st.error(f"Model could not load or predict: {exc}")
    if "batch_rows" in st.session_state:
        rows = st.session_state["batch_rows"]
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download batch CSV", csv_bytes(rows), file_name="batch_sentiment_results.csv", mime="text/csv")

st.divider()
with st.expander("📚 Assignment: Understanding attention"):
    st.markdown("""
**Question 1: What does self-attention do?**

Self-attention lets a Transformer use relationships between words when interpreting a sentence. For example, **not** changes the sentiment of **good** in “not good.”

**Question 2: Why is attention useful in mixed customer reviews?**

It helps the model consider context across a whole review, such as “Delivery was slow, **but** the product is excellent,” where the reviewer mentions both a complaint and a compliment.
""")
st.caption("Powered by Hugging Face Transformers · Pre-trained public model · No API key")
