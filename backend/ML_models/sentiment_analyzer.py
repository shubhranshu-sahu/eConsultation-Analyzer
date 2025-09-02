# backend/ML_models/sentiment_analyzer.py

from transformers import pipeline

class SentimentAnalyzer:
    """
    A class to handle sentiment analysis using a pre-trained FinBERT model.
    
    This class is designed to load the model only once upon initialization,
    making it efficient for use in a web application.
    """
    def __init__(self):
        """
        Initializes the SentimentAnalyzer and loads the FinBERT model.
        The model is downloaded from Hugging Face the first time it's used
        and cached for subsequent runs.
        """
        
        print("Initializing Sentiment Analyzer... This will take a moment.")
        
        try:
            # Loading the pre-trained text classification pipeline from Hugging Face
            
             # --- The Key Change is Here --- :SHUB: adding  parameter ( return_all_scores=True ) to get all scores !!!
            self.sentiment_pipeline = pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                return_all_scores=True  # Tell the pipeline to return scores for all labels
            )
            
            """ 
                UserWarning: `return_all_scores` is now deprecated,  if want a similar functionality use `top_k=None` instead of `return_all_scores=True` or `top_k=1` instead of `return_all_scores=False`.
                
                --------------Change it later ----------------------- :SHUB:

            """
            
            print("Sentiment Analyzer initialized successfully.")
        
        except Exception as e:
            print(f"Error initializing sentiment model: {e}")
            self.sentiment_pipeline = None

    def analyze(self, comment_text: str) -> dict:
        """
        Analyzes a single piece of text for sentiment and returns the
        full score distribution.
        Args:
            comment_text (str): The comment string to be analyzed.

        Returns:
            dict: A dictionary containing the 'sentiment' (e.g., 'Positive') 
                  and the 'score' (confidence of the model).
        """
        if not self.sentiment_pipeline:
            return {"error": "Sentiment model not loaded."}
        
        if not isinstance(comment_text, str) or not comment_text.strip():
            return {"sentiment": "Neutral", "score": 1.0, "info": "Empty or invalid input"}

        try:
            # BERT-based models have a maximum token limit (usually 512).
            # We truncate the text to avoid errors with very long comments.
            truncated_text = comment_text[:512]

            # Run the text through the analysis pipeline
            result = self.sentiment_pipeline(truncated_text)

             # The output is now a list inside a list, e.g., [[{'label': 'negative', 'score': 0.8...}, ...]]
            # We process this to find the top sentiment and include all scores.
            scores = {item['label']: round(item['score'], 4) for item in result[0]}
            top_sentiment = max(scores, key=scores.get)

            return {
                "top_sentiment": top_sentiment.capitalize(),
                "top_score": scores[top_sentiment],
                "all_scores": scores
            }

        except Exception as e:
            print(f"Error during sentiment analysis: {e}")
            return {"error": str(e)}

# --- Example Usage ---
# This block of code allows you to test this file directly
# by running "python backend/ML_models/sentiment_analyzer.py" in your terminal.
if __name__ == '__main__':
    # 1. Create an instance of the analyzer. The model will be downloaded/loaded here.
    analyzer = SentimentAnalyzer()

    # 2. Create a list of example comments to test
    example_comments = [
        "The proposed amendment to Rule 11(2) for Finance Companies is a significant step towards improving the ease of doing business.",
        "I have serious concerns about the potential risks associated with extending these exemptions without proper oversight.",
        "This notice is clear and provides all the necessary details.",
        "The deadline of July 17th is too short for stakeholders to provide meaningful feedback.",
        "" # Empty comment
        , "This is just what not should be done"
    ]

    print("\n--- Testing the Sentiment Analyzer ---")
    # 3. Loop through the comments and analyze each one
    for i, comment in enumerate(example_comments):
        analysis_result = analyzer.analyze(comment)
        print(f"Comment {i+1}: '{comment[:50]}...'")
        print(f"Result: {analysis_result}\n")