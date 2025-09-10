import pandas as pd
from transformers import pipeline
import re
import json

# Ensure document_parser.py is accessible
from backend.data_processing.document_parser import DocumentParser

class AspectSentimentAnalyzer:
    """
    Performs Aspect-Based Sentiment Analysis by linking comments to specific
    sections of a parsed legal document with improved aspect detection logic.
    """
    def __init__(self):
        print("Initializing Aspect-Based Sentiment Analyzer...")
        self.sentiment_pipeline = pipeline("text-classification", model="ProsusAI/finbert", top_k=None)
        self.document_parser = DocumentParser()
        self.parsed_document_content = None
        self.document_aspects = []
        print("Analyzer initialized successfully.")

    def load_and_parse_document(self, pdf_path: str):
        """Loads and parses the notice PDF to be used as context."""
        print(f"Loading and parsing document: {pdf_path}")
        try:
            with open(pdf_path, "rb") as f:
                self.parsed_document_content = self.document_parser.parse(f)
                # --- IMPROVEMENT: Normalize aspect keys for better matching ---
                self.document_aspects = [re.sub(r'[\s\(\).,]+', '', key).lower() for key in self.parsed_document_content.keys()]
                print("Document parsed successfully. Aspects identified and normalized.")
        except FileNotFoundError:
            print(f"ERROR: Document not found at {pdf_path}")
            self.parsed_document_content = None
            self.document_aspects = []

    def _find_aspects_in_text(self, comment_text: str) -> list:
        """Finds aspects in 'Document Level' comments by keyword matching."""
        found_aspects = []
        normalized_comment = re.sub(r'[\s\(\).,]+', '', comment_text).lower()
        for aspect in self.document_aspects:
            if aspect in normalized_comment:
                # Find the original key from the parsed document
                for original_key in self.parsed_document_content.keys():
                    if re.sub(r'[\s\(\).,]+', '', original_key).lower() == aspect:
                        found_aspects.append(original_key)
                        break
        return list(set(found_aspects)) # Use set to get unique aspects

    def analyze_comment(self, comment_row: pd.Series) -> dict:
        """
        Analyzes a single comment for aspect-based sentiment.
        """
        if self.parsed_document_content is None:
            return {"error": "Document context not loaded."}

        comment_text = comment_row['CommentText']
        section_id = comment_row['SectionID']
        
        aspects_to_analyze = []
        if section_id and section_id != "Document Level":
            aspects_to_analyze.append(section_id)
        else:
            aspects_to_analyze = self._find_aspects_in_text(comment_text)

        # Always include an overall sentiment for the full comment
        aspect_sentiments = {"Overall Sentiment": self._get_sentiment_for_text(comment_text)}

        if not aspects_to_analyze:
            return aspect_sentiments

        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', comment_text)
        
        for aspect in aspects_to_analyze:
            for i, sentence in enumerate(sentences):
                normalized_sentence = re.sub(r'[\s\(\).,]+', '', sentence).lower()
                normalized_aspect = re.sub(r'[\s\(\).,]+', '', aspect).lower()
                if normalized_aspect in normalized_sentence:
                    # --- IMPROVEMENT: Analyze a wider context window ---
                    context_window = sentence
                    if i + 1 < len(sentences):
                        context_window += " " + sentences[i+1] # Add the next sentence
                    
                    aspect_sentiments[aspect] = self._get_sentiment_for_text(context_window)
                    break 
        
        return aspect_sentiments

    def _get_sentiment_for_text(self, text: str) -> str:
        """Helper to run sentiment pipeline and return the top label."""
        try:
            result = self.sentiment_pipeline(text[:512])[0]
            scores = {item['label']: item['score'] for item in result}
            return max(scores, key=scores.get).capitalize()
        except Exception:
            return "Error"

# --- Main script to demonstrate the full workflow ---
if __name__ == '__main__':
    NOTICE_PDF_PATH = "C:\\Users\\Vansh\\Desktop\\SIH\\Project\\eConsultation-Analyzer\\backend\\data_processing\\Public-notice-bilingual-languge-20250721.pdf"
    COMMENTS_CSV_PATH = "C:\\Users\\Vansh\\Desktop\\SIH\\Project\\dummy.csv"

    absa_analyzer = AspectSentimentAnalyzer()
    absa_analyzer.load_and_parse_document(NOTICE_PDF_PATH)

    try:
        comments_df = pd.read_csv(COMMENTS_CSV_PATH)
    except FileNotFoundError:
        print(f"ERROR: Comments file not found. Exiting.")
        exit()

    print("\n--- Running Improved ABSA on a Sample of 10 Comments ---")
    sample_df = comments_df#.sample(10, random_state=42) # Use random_state for reproducible sample

    for index, row in sample_df.iterrows():
        print(f"\n--- Analyzing CommentID: {row['CommentID']} ---")
        print(f"Original Comment: {row['CommentText']}")
        
        true_aspects = json.loads(row['multi_value'])
        print(f"Ground Truth Aspects: {true_aspects}")
        
        predicted_aspects = absa_analyzer.analyze_comment(row)
        print(f"Predicted Aspects:    {predicted_aspects}")
