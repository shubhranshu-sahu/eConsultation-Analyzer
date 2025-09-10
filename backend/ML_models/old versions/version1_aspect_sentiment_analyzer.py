# backend/ML_models/aspect_sentiment_analyzer.py

import pandas as pd
from transformers import pipeline
import re
import json

# We will import your existing DocumentParser class
# Ensure document_parser.py is in the backend/data_processing/ folder
from backend.data_processing.document_parser import DocumentParser

class AspectSentimentAnalyzer:
    """
    Performs Aspect-Based Sentiment Analysis by linking comments to specific
    sections of a parsed legal document.
    """
    def __init__(self):
        """Initializes the sentiment model and the document parser."""
        print("Initializing Aspect-Based Sentiment Analyzer...")
        self.sentiment_pipeline = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            top_k=None
        )
        self.document_parser = DocumentParser()
        self.parsed_document_content = None
        print("Analyzer initialized successfully.")

    def load_and_parse_document(self, pdf_path: str):
        """Loads and parses the notice PDF to be used as context."""
        print(f"Loading and parsing document: {pdf_path}")
        try:
            with open(pdf_path, "rb") as f:
                self.parsed_document_content = self.document_parser.parse(f)
                # Get a list of potential aspects (section IDs) from the parsed doc
                self.document_aspects = list(self.parsed_document_content.keys())
                print("Document parsed successfully. Aspects identified.")
        except FileNotFoundError:
            print(f"ERROR: Document not found at {pdf_path}")
            self.parsed_document_content = None
            self.document_aspects = []

    def _find_aspects_in_text(self, comment_text: str) -> list:
        """Finds aspects in 'Document Level' comments by keyword matching."""
        found_aspects = []
        for aspect in self.document_aspects:
            # Use regex to find whole words to avoid partial matches
            if re.search(r'\b' + re.escape(aspect) + r'\b', comment_text, re.IGNORECASE):
                found_aspects.append(aspect)
        return found_aspects

    def analyze_comment(self, comment_row: pd.Series) -> dict:
        """
        Analyzes a single comment (as a row from the DataFrame) for
        aspect-based sentiment.
        """
        if self.parsed_document_content is None:
            return {"error": "Document context not loaded. Call load_and_parse_document() first."}

        comment_text = comment_row['CommentText']
        section_id = comment_row['SectionID']
        
        aspects_to_analyze = []
        if section_id and section_id != "Document Level":
            # The user has explicitly tagged the comment to a section
            aspects_to_analyze.append(section_id)
        else:
            # For general comments, we must find aspects from the text
            aspects_to_analyze = self._find_aspects_in_text(comment_text)

        if not aspects_to_analyze:
            return {"Document Level": self._get_sentiment_for_text(comment_text)}

        # Split comment into sentences for more granular analysis
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', comment_text)
        aspect_sentiments = {}

        for aspect in aspects_to_analyze:
            for sentence in sentences:
                if re.search(r'\b' + re.escape(aspect) + r'\b', sentence, re.IGNORECASE):
                    aspect_sentiments[aspect] = self._get_sentiment_for_text(sentence)
                    break # Analyze the first sentence that mentions the aspect
        
        # If no specific aspect sentence was found, fall back to general sentiment
        if not aspect_sentiments:
             return {"Document Level": self._get_sentiment_for_text(comment_text)}

        return aspect_sentiments

    def _get_sentiment_for_text(self, text: str) -> str:
        """Helper function to run the sentiment pipeline and return the top label."""
        try:
            result = self.sentiment_pipeline(text[:512])[0]
            scores = {item['label']: item['score'] for item in result}
            return max(scores, key=scores.get).capitalize()
        except Exception:
            return "Error"

# --- Main script to demonstrate the full workflow ---
if __name__ == '__main__':
    # Define paths to your files
    NOTICE_PDF_PATH = "C:\\Users\\Vansh\\Desktop\\SIH\\Project\\eConsultation-Analyzer\\backend\\data_processing\\Public-notice-bilingual-languge-20250721.pdf"
    COMMENTS_CSV_PATH = "C:\\Users\\Vansh\\Desktop\\SIH\\Project\\dummy.csv"

    # 1. Initialize the analyzer
    absa_analyzer = AspectSentimentAnalyzer()
    
    # 2. Load and parse the source document to get context
    absa_analyzer.load_and_parse_document(NOTICE_PDF_PATH)

    # 3. Load the comments dataset
    try:
        comments_df = pd.read_csv(COMMENTS_CSV_PATH)
        print(f"\nLoaded {len(comments_df)} comments from '{COMMENTS_CSV_PATH}'.")
    except FileNotFoundError:
        print(f"ERROR: Comments file not found at '{COMMENTS_CSV_PATH}'. Exiting.")
        exit()

    print("\n--- Running ABSA on a Sample of 10 Comments ---")
    
    # Take a sample of 10 rows to test
    sample_df = comments_df#.sample(10)

    for index, row in sample_df.iterrows():
        print(f"\n--- Analyzing CommentID: {row['CommentID']} ---")
        print(f"Original Comment: {row['CommentText']}")
        
        # This is the ground truth from your dataset for comparison
        true_aspects = json.loads(row['multi_value'])
        print(f"Ground Truth Aspects: {true_aspects}")
        
        # Run the analysis
        predicted_aspects = absa_analyzer.analyze_comment(row)
        print(f"Predicted Aspects:    {predicted_aspects}")