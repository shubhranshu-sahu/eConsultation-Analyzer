# backend/ML_models/summarizer.py

from transformers import pipeline

class Summarizer:
    """
    A class to handle text summarization using a pre-trained BART model.
    This class is designed to load the model only once upon initialization.
    """
    def __init__(self):
        """
        Initializes the Summarizer and loads the BART model.
        This is a large model (~1.6 GB) and will take time to download
        and load for the first time.
        """
        print("Initializing Summarizer (facebook/bart-large-cnn)... This is a large model and may take a while.")
        try:
            # Load the pre-trained summarization pipeline from Hugging Face
            self.summarization_pipeline = pipeline(
                "summarization",
                model="facebook/bart-large-cnn"
            )
            print("Summarizer initialized successfully.")
        except Exception as e:
            print(f"Error initializing summarization model: {e}")
            self.summarization_pipeline = None

    def summarize(self, comment_text: str, min_length: int = 20, max_length: int = 60) -> str:
        """
        Generates a summary for a single piece of text.

        Args:
            comment_text (str): The comment string to be summarized.
            min_length (int): The minimum number of words for the summary.
            max_length (int): The maximum number of words for the summary.

        Returns:
            str: The generated summary string.
        """
        if not self.summarization_pipeline:
            return "Error: Summarization model not loaded."
        
        if not isinstance(comment_text, str) or len(comment_text.split()) < min_length:
            return comment_text # Return original text if it's too short to summarize

        try:
            # BERT/BART models have a token limit (usually 1024). Truncate to be safe.
            truncated_text = ' '.join(comment_text.split()[:1024])

            # Run the text through the summarization pipeline with fine-tuned parameters
            result = self.summarization_pipeline(
                truncated_text,
                min_length=min_length,
                max_length=max_length,
                num_beams=4,          # Use beam search for higher quality
                length_penalty=2.0,   # Encourage model to not stop too early
                early_stopping=True
            )
            
            return result[0]['summary_text']

        except Exception as e:
            print(f"Error during summarization: {e}")
            return "Error: Could not generate summary."

# --- Example Usage ---
# This block allows you to test this file directly
if __name__ == '__main__':
    # 1. Create an instance of the summarizer. The model will be downloaded/loaded here.
    summarizer_instance = Summarizer()

    # 2. A long, detailed example comment
    long_comment = """
    While I appreciate the Ministry's goal of enhancing the ease of doing business for Finance Companies
    within the IFSC jurisdiction, I believe the current draft of the amendment to Rule 11(2) is not
    without its significant flaws. The proposal to extend the exemption currently available to NBFCs
    registered with the RBI is logical on the surface, but it fails to account for the different regulatory
    and risk profiles of these new entities. Simply broadening the exemption without introducing
    commensurate supervisory mechanisms could create a regulatory gap, potentially undermining the
    financial stability that the primary regulations are designed to protect. We strongly suggest that the
    Ministry consider a phased rollout or introduce additional criteria that these Finance Companies must
    meet before being granted such a powerful exemption.
    """

    print("\n--- Testing the Summarizer ---")
    
    # 3. Generate a summary with default settings
    summary = summarizer_instance.summarize(long_comment)
    print("\nGenerated Summary (default settings):")
    print(f"-> {summary}")

    # 4. Generate a shorter, more concise summary
    short_summary = summarizer_instance.summarize(long_comment, min_length=15, max_length=40)
    print("\nGenerated Summary (shorter settings):")
    print(f"-> {short_summary}")