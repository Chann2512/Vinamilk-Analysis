import os
import sys
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from pyvi import ViTokenizer
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, f1_score

# Ensure the environment settings (redirects, timeout, endpoints) are applied
import utils

def balance_dataset(df, random_state=42):
    """
    Down-samples the majority Positive class (label 2) to around 1400 rows
    to match the minority classes and shuffles the dataset.
    """
    print("Balancing dataset via down-sampling...")
    df_0 = df[df['LLM_Sentiment'] == 0]
    df_1 = df[df['LLM_Sentiment'] == 1]
    df_2 = df[df['LLM_Sentiment'] == 2]
    
    # Check shape to avoid ValueError if df_2 has fewer rows (unlikely)
    sample_size = min(1400, len(df_2))
    df_2_sampled = df_2.sample(n=sample_size, random_state=random_state)
    
    df_balanced = pd.concat([df_0, df_1, df_2_sampled], axis=0)
    df_balanced = df_balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    print(f"Original shape: {len(df)} -> Balanced shape: {len(df_balanced)}")
    print("Class distribution:\n", df_balanced['LLM_Sentiment'].value_counts())
    return df_balanced

def tokenize_func(text):
    """
    Applies pyvi word tokenizer to format Vietnamese text for PhoBERT (joins compound words).
    """
    try:
        return ViTokenizer.tokenize(str(text))
    except Exception:
        return str(text)

def compute_metrics(eval_pred):
    """
    Computes accuracy and macro F1 score for evaluation.
    """
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {
        "accuracy": acc,
        "f1_macro": f1
    }

def train_model(data_path, output_dir, model_name="vinai/phobert-base", epochs=3, batch_size=4):
    """
    Full training pipeline:
    1. Loads classified reviews
    2. Performs down-sampling class balancing
    3. Train-test split (80-20 stratified)
    4. Segments Vietnamese words
    5. Converts to Hugging Face datasets and tokenizes
    6. Fine-tunes PhoBERT model and saves results
    """
    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path, encoding="utf-8-sig")
    
    # Drop rows without labels or comments
    df = df.dropna(subset=['LLM_Sentiment', 'comment'])
    df['LLM_Sentiment'] = df['LLM_Sentiment'].astype(int)
    
    # 1. Class Balancing
    df_balanced = balance_dataset(df)
    
    # 2. Train-Test Split (80/20 Stratified)
    train_df, test_df = train_test_split(
        df_balanced,
        test_size=0.2,
        random_state=42,
        stratify=df_balanced['LLM_Sentiment']
    )
    
    # 3. Vietnamese Word Segmentation
    print("Performing word segmentation using ViTokenizer (PyVi)...")
    train_df['text'] = train_df['comment'].apply(tokenize_func)
    test_df['text'] = test_df['comment'].apply(tokenize_func)
    
    # 4. Hugging Face Dataset Conversion
    print("Converting to Hugging Face datasets...")
    train_dataset = Dataset.from_pandas(train_df[['text', 'LLM_Sentiment']].reset_index(drop=True))
    test_dataset = Dataset.from_pandas(test_df[['text', 'LLM_Sentiment']].reset_index(drop=True))
    
    # Rename label column to match Trainer specifications
    train_dataset = train_dataset.rename_column("LLM_Sentiment", "label")
    test_dataset = test_dataset.rename_column("LLM_Sentiment", "label")
    
    # 5. Tokenization
    print(f"Downloading/loading tokenizer for: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_batch(batch):
        return tokenizer(batch["text"], truncation=True, max_length=256)
    
    train_dataset = train_dataset.map(tokenize_batch, batched=True)
    test_dataset = test_dataset.map(tokenize_batch, batched=True)
    
    # Set format for PyTorch
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    
    # 6. Initialize Model
    print(f"Downloading/loading pre-trained model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    
    # Hardware check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training hardware: {device.upper()}")
    if device == "cuda":
        print(f"   + GPU Name: {torch.cuda.get_device_name(0)}")
    else:
        print("   WARNING: No GPU found. Local CPU training will be extremely slow!")
        
    # 7. Configure Training Arguments
    # Dynamic fp16 setting: Must be False on CPU to avoid crashes
    use_fp16 = torch.cuda.is_available()
    
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        fp16=use_fp16,
        logging_steps=100,
        report_to="none" # Disable integrations to prevent warning messages
    )
    
    # Data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )
    
    # Run training
    print("Starting fine-tuning...")
    trainer.train()
    
    # Save best model and tokenizer
    print(f"Saving final model and tokenizer to: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Training pipeline completed successfully!")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_data = os.path.join(project_root, "data", "raw", "classified_reviews.csv")
    default_output = os.path.join(project_root, "models", "phobert_sentiment")
    
    train_model(default_data, default_output)
