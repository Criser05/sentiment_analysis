# Financial News Sentiment Analysis FinBERT Fine-Tuning

**3-class sentiment classification** (positive / neutral / negative) on financial news sentences.  
Pipeline: TF-IDF + Logistic Regression baseline → fine-tuning of `ProsusAI/FinBERT` via HuggingFace Transformers.


---

## Results

| Model | Vectorisation | F1-Macro |
|---|---|---|
| Logistic Regression | TF-IDF unigrams | ~0.79 |
| Logistic Regression | TF-IDF bigrams + `class_weight='balanced'` | 0.82 |
| **FinBERT fine-tuned** | Contextual BERT embeddings | **0.934** |

Fine-tuning FinBERT yields a **+11.4 point gain in F1-Macro** over the best TF-IDF baseline.

---

## Dataset

**Financial PhraseBank** (Malo et al., 2014) — 2,264 sentences extracted from financial news, manually annotated by domain experts into three sentiment classes.

- Loaded via `FinanceMTEB/financial_phrasebank` (HuggingFace Hub)
- Pre-split into train and test sets
- 1 duplicate removed from train before any processing
- **Class distribution:** imbalanced — neutral dominates, motivating F1-Macro over accuracy as the primary metric

---

## Pipeline

### 1. Baseline — TF-IDF + Logistic Regression

Two variants compared:

**Unigrams (default):** each word treated independently. Key failure mode — negations are lost (`"not good"` → `"not"` + `"good"`, the model reads `"good"` as positive). Consequence: low recall on the negative class, which is the most costly error in a financial context a missed negative signal can mean failing to exit a losing position in time.

**Bigrams + balanced weights:** extending to `ngram_range=(1,2)` preserves two-word expressions and captures negation patterns. Adding `class_weight='balanced'` corrects for the neutral-class dominance. This combination yields **+0.21 recall on the negative class** over the unigram baseline.

Despite these improvements, TF-IDF represents text as a bag of words so word order and context are discarded. `"strong growth"` and `"no strong growth"` share identical unigram representations.

### 2. FinBERT Fine-Tuning

`ProsusAI/FinBERT` is a BERT-base model pre-trained on financial corpora (10-K filings), giving it prior domain knowledge that general-purpose models lack.

**Preprocessing:**
- Label column cast to `ClassLabel` (required by HuggingFace Trainer for stratified splits and metric computation)
- Stratified 80/20 train/validation split (seed=42)
- Tokenization with `truncation=True` no sequences exceed 150 tokens in this dataset (BERT limit: 512), so truncation has no practical effect
- Dynamic padding via `DataCollatorWithPadding` sequences padded to the longest in each batch, not globally, reducing memory usage

**Training setup:**

| Hyperparameter | Value |
|---|---|
| Base model | `ProsusAI/finbert` |
| Epochs | 3 |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Eval strategy | Per epoch |

**Metrics tracked:** F1-Macro and Recall-Macro both penalise poor performance on minority classes.

### 3. Error Analysis

Confusion matrix analysis reveals that the majority of FinBERT's remaining errors are positive sentences misclassified as negative (18 cases). Many of these are genuinely ambiguous sentences reporting marginal year-on-year improvements that the annotators labelled positive but that could reasonably signal stagnation. Crucially, **recall on the negative class remains high**, meaning the model reliably catches true negative signals the highest-stakes error category in financial applications.

---

## Repository Structure

```
finbert-financial-sentiment/
├── finbert_sentiment_analysis.ipynb   # Full pipeline notebook
├── requirements.txt
└── README.md
```

## How to Run
1. Open the notebook in Google Colab
2. Enable GPU runtime (T4)
3. Run all cells 

> **Data:** loaded automatically from HuggingFace Hub (`FinanceMTEB/financial_phrasebank`). No local files required.

---

## Stack

**Language:** Python  
**Libraries:** `transformers`, `datasets`, `evaluate`, `scikit-learn`, `torch`, `pandas`, `numpy`, `matplotlib`, `seaborn`  
**Methods:** TF-IDF (unigrams / bigrams), Logistic Regression, FinBERT fine-tuning (HuggingFace Trainer API), F1-Macro evaluation, confusion matrix analysis

---

## Reference

Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014). *Good debt or bad debt: Detecting semantic orientations in economic texts.* Journal of the American Society for Information Science and Technology, 65(4), 782–796.  
Model: [`ProsusAI/finbert`](https://huggingface.co/ProsusAI/finbert)

