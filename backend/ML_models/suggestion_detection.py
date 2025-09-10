# backend/ML_models/suggestion_detector.py


from transformers import pipeline

class SuggestionDetector:
    """
    A class to detect actionable suggestions in text using a
    Zero-Shot Classification model.
    """
    def __init__(self):
        """
        Initializes the detector by loading the Zero-Shot model.
        Note: This is a large model and may be loaded by another module.
        In a full app, you'd share one instance to save memory.
        """
        print("Initializing Suggestion Detector (Zero-Shot Model)...")
        try:
            self.classifier = pipeline(
                "zero-shot-classification",
                model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
            )
            print("Suggestion Detector initialized successfully.")
        except Exception as e:
            print(f"Error initializing Zero-Shot model: {e}")
            self.classifier = None

    def contains_suggestion(self, comment_text: str, threshold: float = 0.60) -> bool:
        """
        Classifies if a comment contains a suggestion based on its intent.

        Args:
            comment_text (str): The user's comment.
            threshold (float): The confidence score required to classify as a suggestion.

        Returns:
            bool: True if the comment is classified as a suggestion, False otherwise.
        """
        if not self.classifier or not isinstance(comment_text, str) or not comment_text.strip():
            return False

        # The two categories we want the model to choose between
        candidate_labels = ["opinion or feedback", "suggestion for a change"]

        try:
            # Truncate text to be safe with model limits
            truncated_text = comment_text[:512]
            
            result = self.classifier(truncated_text, candidate_labels=candidate_labels, multi_label=False)
            
            top_label = result['labels'][0]
            top_score = result['scores'][0]

            # Check if the top label is the one we're looking for AND if the model is confident
            if top_label == candidate_labels[1] and top_score > threshold:
                return True
            
            return False
            
        except Exception as e:
            print(f"Error during suggestion detection: {e}")
            return False

# --- Example Usage (for standalone testing) ---
if __name__ == '__main__':
    # Create an instance of the detector. This will download/load the model.
    detector = SuggestionDetector()

    comment_with_suggestion = "The current draft is flawed. The Ministry should consider adding a clause to protect small businesses."
    comment_without_suggestion = "I am deeply concerned about the potential risks of this new rule."
    borderline_comment = "This proposal could be improved."

    print("\n--- Testing the Suggestion Detector ---")
    is_suggestion1 = detector.contains_suggestion(comment_with_suggestion)
    print(f"Comment 1 ('{comment_with_suggestion[:40]}...') contains suggestion: {is_suggestion1}") # Expected: True

    is_suggestion2 = detector.contains_suggestion(comment_without_suggestion)
    print(f"Comment 2 ('{comment_without_suggestion[:40]}...') contains suggestion: {is_suggestion2}") # Expected: False
    
    is_suggestion3 = detector.contains_suggestion(borderline_comment)
    print(f"Comment 3 ('{borderline_comment[:40]}...') contains suggestion: {is_suggestion3}") # Expected: True