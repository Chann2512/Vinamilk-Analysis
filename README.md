# 📊 Shopee Sentiment Analysis & Aspect Classification Project

This repository contains a modularized, production-ready implementation of a multi-aspect sentiment analysis pipeline optimized for E-commerce data. 

Focusing on **Vinamilk's Official Store on Shopee Mall**, this project leverages state-of-the-art Natural Language Processing (NLP) frameworks to transform unstructured consumer reviews into actionable corporate strategy. The codebase has been fully reorganized from monolithic notebook logic into clean, reusable Python modules (`src/`) and sequential notebooks (`notebooks/`).

---

## 🏢 Business Context: The Vinamilk E-Commerce Challenge

**Vinamilk** is the leading national brand in the dairy sector. As consumer behavior shifts dynamically toward digital channels, managing brand health and customer experience on e-commerce platforms like Shopee is critical. 

### The Problem:
* **The "Coin-Reward Blindspot":** E-commerce datasets suffer from extreme **Positive Bias**. Shopee’s reward system incentivizes buyers to instantly rate 5-stars with generic phrases (*"good product"*, *"fast delivery"*) to earn loyalty coins. 
* **Hidden Friction Points:** Standard aggregate metrics (such as Star Ratings) completely mask underlying operational flaws. If a customer is furious about a dented milk carton or rude customer service but still clicks 5-stars for rewards, traditional business intelligence treats it as a perfect transaction.

### The Objective:
To build an end-to-end NLP pipeline that bypasses the "star-rating bias" by deeply analyzing the **semantic context of review texts**. The system automatically classifies text into **5 core business aspects** and **3 sentiment polarities**, exposing hidden operational bottlenecks.

---

## ⚙️ Core Architecture & NLP Framework

The system utilizes an advanced **Hate-Speech & Token-based preprocessing stack** combined with **PhoBERT (Pre-trained Language Model for Vietnamese)** to achieve granular textual understanding.

This repository contains the modularized, production-ready implementation of the Shopee Sentiment Analysis and Aspect classification pipeline. It reorganizes the original notebook logic into clean, reusable Python modules (`src/`) and step-by-step Jupyter Notebooks (`notebooks/`) for easy profiling, cleaning, filtering, and model training.

---

## 📂 Repository Structure

The project directory structure is laid out as follows:

```
Shopee-Sentiment-Analysis/
│
├── README.md               # Project guide and execution documentation
├── requirements.txt        # Core project dependencies list
├── .gitignore              # Files and folders to exclude from version control
│
├── data/
│   ├── raw/                # Unmodified raw source data files
│   │   ├── shopee_reviews.csv      # Extracted raw reviews sheet
│   │   └── classified_reviews.csv  # LLM-gán nhãn aspect/sentiment results
│   │
│   └── processed/          # Intermediate processed datasets
│       ├── text_reviews.csv   # Non-empty review text segments
│       ├── clean_reviews.csv  # Standardized normalized reviews
│       ├── valid_reviews.csv  # Validated reviews (after filtering out spam)
│       └── train_dataset.csv  # Final preprocessed training dataset
│
├── src/                    # Reusable Python source code modules
│   ├── utils.py            # Local cache redirections and environment checkups
│   ├── preprocessing.py    # Text cleaning, normalization, and slang translation
│   ├── filtering.py        # Length, Shannon Entropy, and keyword density filtering
│   ├── labeling.py        # Rating star mapping and aspect spelling correction
│   ├── train.py            # Dataset down-sampling, PyTorch mapping, and training loop
│   └── predict.py          # Prediction class and inference wraps
│
├── notebooks/              # Step-by-step pipeline Jupyter Notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_review_filtering.ipynb
│   ├── 04_sentiment_labeling.ipynb
│   ├── 05_phobert_training.ipynb
│   └── 06_evaluation.ipynb
│
├── models/
│   └── phobert_sentiment/  # Saved checkpoint weights of fine-tuned PhoBERT
│
└── outputs/
    ├── figures/            # Visual chart plots (Aspect, Sentiment distributions)
    ├── reports/            # Exported evaluation reports
    └── predictions.csv     # Extracted predictions
```

---

## 🛠️ Installation & Setup

1. **Activate Virtual Environment** (Make sure python is installed):
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

2. **Install Core Dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

3. **Verify Python Pathing**:
   Ensure `src/` functions are discoverable by running the test script.

---

## ⚙️ Python Modules in `src/`

| Module | Description | Core Functions |
|---|---|---|
| **`utils.py`** | Environmental settings, Hugging Face cache redirections, and mirrors. | `setup_environment()`, `ensure_project_dirs()` |
| **`preprocessing.py`** | Text normalization, cleaning, URL/emoji removal, and slang translation. | `normalize_unicode()`, `remove_emoji()`, `replace_slang()`, `clean_text()` |
| **`filtering.py`** | Shannon Entropy calculation, dictionary validation, and filtering. | `shannon_entropy()`, `dictionary_ratio()`, `apply_filters()` |
| **`labeling.py`** | Star-to-label mapping and LLM aspect typo corrections. | `map_star_to_label()`, `correct_aspect_typos()`, `load_classified_dataset()` |
| **`train.py`** | Down-sampling class balancer, training parameters, and training loop. | `balance_dataset()`, `train_model()`, `compute_metrics()` |
| **`predict.py`** | Device routing (GPU/CPU) and custom prediction wrappers. | `SentimentPredictor.predict()`, `SentimentPredictor.predict_batch()` |

---

## 🔄 Notebook Execution Sequence

Run the notebooks sequentially under the `notebooks/` directory:

1. **`01_data_exploration.ipynb`**: Handles dataset loading, schema checks, missing values cleaning, and divides the dataset into star-only reviews and text reviews.
2. **`02_preprocessing.ipynb`**: Performs Unicode NFC mapping, emoji/URL stripping, compressing double characters, and translating abbreviations.
3. **`03_review_filtering.ipynb`**: Computes Shannon entropy, filters spam and non-relevant reviews, and outputs the clean validation subset.
4. **`04_sentiment_labeling.ipynb`**: Cleans aspect spelling errors, maps sentiment labels, and formats files for training.
5. **`05_phobert_training.ipynb`**: Performs down-sampling for class balance and starts the PhoBERT sequence-classification training loop.
6. **`06_evaluation.ipynb`**: Renders charts, draws aspect distributions, plots the aspect-sentiment heatmap, and tracks timeline trend changes for business insight.

---

## 📊 Analytical Insights (Key Findings)

Deploying this pipeline across **~120,000 raw customer interactions** yielded critical corporate intelligence:

* **Product Quality Resilience:** The **"Quality"** (Chất lượng) and **"Product Match"** (Đúng mô tả) aspects maintain an absolute positive sentiment rate of **99.3% and 99.5%** respectively. Vinamilk's core product line holds impeccable consumer trust.
* **The Logistics Vulnerability:** Despite an overwhelming overall positive bias, the **"Delivery"** (Giao hàng) and **"Customer Service"** (Dịch vụ) dimensions captured the **highest negative sentiment concentrations (1.3% and 1.1%)**. 
* **Time-Series Volatility & Seasonality:** Granular time-series evaluations tracked a massive dreg in interaction volume during February ("The Tet Slack"), followed by a shaped exponential spike in April, aligning directly with major platform Mega-Sale cycles (e.g., 3.3, 4.4). Crucially, even when transaction volumes multiplied tenfold, the negative sentiment baseline remained controlled, validating supply chain elasticity.
---

## 📈 Executive Insights & Actionable Strategy

Deploying this pipeline across the analytical framework generated key operational insights:

* **Insight 1 - Core Trust Stability:** Product Quality acts as the main positive driver (99.3% positive baseline). Brand retention is anchored strongly in flavor consistency and nutritional trust.
* **Insight 2 - Delivery Vulnerabilities:** Negative feedback spikes primarily within **Logistics (1.3% negative cluster)** and **Service (1.1% negative cluster)** due to dented boxes and package leaks caused during transit.
* **Insight 3 - Promotion Seasonality:** Sentiment trends exhibit strict seasonality. Volummetric interaction climbs significantly during Mega-Sale days (11.11, 12.12), dropping abruptly during February ("The Tet Slack").

---

## 🎯 Strategic Action Roadmap

### 📦 Short-Term Operations
* Tighten Third-Party Logistics (3PL) auditing on Shopee to mitigate package denting.
* Integrate dynamic bubble-wraps on liquid milk packages during peak rain seasons.

### 📊 Mid-Term Scaling
* Implement automatic real-time sentiment tracking dashboards.
* Scale customer service staff up by 40% precisely 48 hours post-Mega Sale windows.

### 💼 Long-Term Vision
* Embed sentiment engine pipelines directly into corporate CRM architecture.
* Drive product roadmap and consumer research decisions using continuous Voice-of-Customer (VoC) analytics.

---

## 🎯 Strategic Business Actions Recommended

Based on the Sentiment-Aspect Matrix and Time-Series trends, Vinamilk's E-commerce division should execute the following operations:

1. **Logistics Partner SLA Overhaul:** Since negative sentiments are decoupled from product quality and concentrated heavily on "Delivery", Vinamilk must enforce stricter Service Level Agreements (SLAs) on Shopee’s 3PL (Third-Party Logistics) providers regarding parcel denting, liquid leaking, and transit delays.
2. **Dynamic Customer Service Allocation:** Customer service friction scales tightly with Shopee's monthly campaign days. Live-chat support staff allocations should scale dynamically up by 40% exactly 48 hours post-Mega Sale events to handle shipping and post-purchase inquiries, suppressing the "Dịch vụ" negative index.
3. **Packaging Optimization:** A key keyword cluster discovered in the negative delivery spectrum relates to damaged packaging. Moving from standard cardboard wraps to higher-grade shock-absorption buffers for milk cartons during peak rain or high-volume sales periods will directly eliminate structural complaints.

---
