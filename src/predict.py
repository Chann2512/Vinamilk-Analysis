import os
import sys
import torch
import torch.nn.functional as F
from pyvi import ViTokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Ensure environments are configured
import utils

# Mapping numeric classes back to Vietnamese string labels
LABEL_MAP = {
    0: "Tiêu cực",
    1: "Trung lập",
    2: "Tích cực"
}

class SentimentPredictor:
    def __init__(self, model_dir=None):
        """
        Loads the fine-tuned PhoBERT model and tokenizer from the model directory.
        """
        if model_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_dir = os.path.join(project_root, "models", "phobert_sentiment")
            
        print(f"Loading sentiment model from: {model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        
        # Route model to appropriate hardware (GPU/CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        print(f"Predictor initialized on device: {str(self.device).upper()}")

    def predict(self, text):
        """
        Predicts sentiment for a single string.
        Returns (predicted_label, probability_list)
        """
        if not isinstance(text, str) or not text.strip():
            return "Trung lập", [0.0, 1.0, 0.0]
            
        # Segment word tokens
        segmented_text = ViTokenizer.tokenize(text)
        
        # Tokenize and move tensors to the correct device
        inputs = self.tokenizer(
            segmented_text,
            return_tensors="pt",
            truncation=True,
            max_length=256
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            
        # Move probabilities to CPU and convert to list
        prob_list = probs.squeeze().cpu().tolist()
        # Handle shape if input text is very short/empty
        if not isinstance(prob_list, list):
            prob_list = [prob_list]
            
        return LABEL_MAP[pred_class], prob_list

    def predict_batch(self, texts, batch_size=32):
        """
        Predicts sentiment for a list of strings in batches.
        Returns a list of predicted labels and list of probability outputs.
        """
        results = []
        probs_out = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            segmented_batch = [ViTokenizer.tokenize(str(t)) if isinstance(t, str) else "" for t in batch_texts]
            
            inputs = self.tokenizer(
                segmented_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=1)
                pred_classes = torch.argmax(probs, dim=1).cpu().tolist()
                batch_probs = probs.cpu().tolist()
                
            results.extend([LABEL_MAP[c] for c in pred_classes])
            probs_out.extend(batch_probs)
            
        return results, probs_out

if __name__ == "__main__":
    # Test block
    try:
        predictor = SentimentPredictor()
        test_sentences = [
            "Sữa thơm ngon béo ngậy, bé nhà mình rất thích uống.",
            "Giao hàng siêu chậm, hộp sữa bị móp méo rách vỏ.",
            "Mới mua về chưa khui hộp nên chưa đánh giá được gì nhiều."
        ]
        print("\n--- Running Quick Predictions ---")
        for s in test_sentences:
            label, probs = predictor.predict(s)
            print(f"Text: '{s}'")
            print(f"Prediction: {label} (Probabilities: Neg={probs[0]:.4f}, Neu={probs[1]:.4f}, Pos={probs[2]:.4f})\n")
    except Exception as e:
        print(f"Skipping quick test execution: Model not fine-tuned yet or {e}")
