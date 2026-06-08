# Shopee Sentiment Analysis Project Reorganization

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
