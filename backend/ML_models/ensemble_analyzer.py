from transformers import pipeline

class EnsembleSentimentAnalyzer:
    """
    An advanced analyzer that uses an ensemble of two models:
    1. A specialist (FinBERT) for domain-specific text.
    2. A generalist (DistilBERT) for common language patterns.
    """
    def __init__(self):
        print("Initializing Ensemble Analyzer... This will take longer as two models are being loaded.")
        try:
            # Load the specialist model (FinBERT)
            self.specialist_pipeline = pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                return_all_scores=True
            )
            
            # Load the generalist model (DistilBERT)
            self.generalist_pipeline = pipeline(
                "sentiment-analysis", # Note: this pipeline is slightly different
                model="distilbert-base-uncased-finetuned-sst-2-english",
                return_all_scores=True
            )
            print("Ensemble Analyzer initialized successfully.")
        except Exception as e:
            print(f"Error initializing ensemble models: {e}")
            self.specialist_pipeline = None
            self.generalist_pipeline = None

    def analyze(self, comment_text: str) -> dict:
        """
        Analyzes a comment using both models and a decision logic to get a more robust prediction.
        """
        if not self.specialist_pipeline or not self.generalist_pipeline:
            return {"error": "Ensemble models not loaded."}
        
        if not isinstance(comment_text, str) or not comment_text.strip():
            return {"error": "Empty or invalid input"}

        try:
            truncated_text = comment_text[:512]

            # --- Get predictions from both models ---
            specialist_result = self.specialist_pipeline(truncated_text)[0]
            generalist_result = self.generalist_pipeline(truncated_text)[0]
            
            # --- Process results into a consistent format ---
            specialist_scores = {item['label'].capitalize(): item['score'] for item in specialist_result}
            specialist_top = max(specialist_scores, key=specialist_scores.get)

            # DistilBERT uses 'POSITIVE'/'NEGATIVE' labels. We'll add Neutral for consistency.
            generalist_scores = {item['label'].capitalize(): item['score'] for item in generalist_result}
            if 'Neutral' not in generalist_scores:
                generalist_scores['Neutral'] = 0.0
            generalist_top = max(generalist_scores, key=generalist_scores.get)

            # --- Ensemble Decision Logic ---
            final_sentiment = specialist_top # Default to the specialist
            
            # 1. If they agree, we are confident.
            if specialist_top == generalist_top:
                final_sentiment = specialist_top
            # 2. If they disagree, trust the one with the higher confidence score.
            else:
                if specialist_scores[specialist_top] > generalist_scores[generalist_top]:
                    final_sentiment = specialist_top
                else:
                    final_sentiment = generalist_top
            
            return {
                "final_sentiment": final_sentiment,
                "specialist_prediction": specialist_top,
                "specialist_scores": {k: round(v, 4) for k, v in specialist_scores.items()},
                "generalist_prediction": generalist_top,
                "generalist_scores": {k: round(v, 4) for k, v in generalist_scores.items()}
            }

        except Exception as e:
            print(f"Error during ensemble analysis: {e}")
            return {"error": str(e)}

if __name__ == '__main__':
    analyzer = EnsembleSentimentAnalyzer()

    # Test the tricky comment that failed before
    tricky_comment = "This is just what not should be done."
    result = analyzer.analyze(tricky_comment)
    
    print(f"\n--- Testing Tricky Comment ---")
    print(f"Comment: '{tricky_comment}'")
    print("\nFull Analysis:")
    import json
    print(json.dumps(result, indent=2))
    print(f"\nFinal Decision: {result.get('final_sentiment')}")