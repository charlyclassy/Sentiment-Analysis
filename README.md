

## 🚀 Live Application

[Launch ReviewSense AI](https://sentiment-analysis-nwprbsteo5fsesp4lgfkkp.streamlit.app/))

Classify customer reviews using a pre-trained
DistilBERT Transformer with Hugging Face.

# 💬 ReviewSense AI — Customer Review Sentiment Analysis

A beginner-friendly NLP project that classifies customer reviews as **POSITIVE** or **NEGATIVE** using a publicly available, pre-trained **DistilBERT Transformer** and Hugging Face `pipeline()`.

## 🎯 Assignment objective

Classify five sample customer reviews using a pre-trained Transformer, display predicted sentiment labels and confidence scores, and compare predictions to expected labels. No API key or model training is required.

## ✨ Features

- Five built-in sample customer reviews with expected labels
- Predicted sentiment and confidence (%) for each review
- Side-by-side comparison and CSV export
- Single-review input and multi-review batch mode
- Dark navy-and-purple Streamlit UI
- Two beginner questions and answers explaining self-attention

## 🧠 Model

- **Model:** [`distilbert/distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)
- **Task:** Binary English-language sentiment classification
- **Labels:** `POSITIVE`, `NEGATIVE`
- **Inference:** CPU (no GPU or API key needed)

This is a fine-tuned, pre-trained model. Confidence is the model's classification score, **not guaranteed accuracy**. It does not output a neutral label. The first run downloads the model from Hugging Face.

## 🛠️ Run in VS Code

Open this folder in VS Code. From **Terminal → New Terminal**:

```bash
python -m venv .venv
```

Activate the environment:

- **Windows PowerShell:** `.venv\Scripts\Activate.ps1`
- **macOS/Linux:** `source .venv/bin/activate`

Install requirements:

```bash
python -m pip install -r requirements.txt
```

Run the Streamlit dashboard:

```bash
python -m streamlit run app.py
```

Your browser will open the local app, usually at `http://localhost:8501`.

## 🧪 Five customer reviews

| # | Review | Expected label |
|---|---|---|
| 1 | I absolutely love this product! It works perfectly. | POSITIVE |
| 2 | The product arrived broken. Very disappointing. | NEGATIVE |
| 3 | Excellent quality and fast delivery. Highly recommended! | POSITIVE |
| 4 | Terrible customer service. I want a refund. | NEGATIVE |
| 5 | Delivery was slow, but the product is really good. | POSITIVE |

Select **Test five sample reviews** and click **Analyse five reviews** to see **actual predictions and confidence scores**. Download the result as `sentiment_results.csv`.

**Important:** Expected labels above are manually assigned. The app computes predictions at runtime; it does not invent or hard-code output labels or confidence percentages. Agreement on five examples is not general model accuracy.

## 📚 Questions about attention

**Q1. What is self-attention in a Transformer?**

Self-attention helps the Transformer interpret words in relation to other words in a sentence. For example, “not” changes the meaning of “good” in “not good.”

**Q2. Why is attention important for customer review sentiment?**

It helps the model use sentence context, including mixed views such as “Delivery was slow, but the product is excellent,” when predicting overall sentiment.

## ☁️ Deploy on Streamlit Community Cloud

1. Upload `app.py`, `requirements.txt`, `README.md`, and `.streamlit/config.toml` to a GitHub repository.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Select the GitHub repository, branch, and `app.py` as the entrypoint.
4. Deploy. The server downloads the model on its first startup. CPU-only inference can be slower than local runs.

## 📁 Folder structure

```text
Customer_Review_Sentiment_Streamlit/
├── app.py
├── requirements.txt
├── README.md
└── .streamlit/
    └── config.toml
```

## 📝 Conclusion

This project demonstrates transfer learning in NLP: a small, pre-trained Transformer performs sentiment classification without training a new neural network. The five-review experiment illustrates labels, model confidence, and the role of context and self-attention. Results depend on the wording of each review and should be validated before real business use.
