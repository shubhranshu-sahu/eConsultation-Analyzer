import pandas as pd
from transformers import pipeline
import warnings

# Suppress warnings that are not critical for functionality
warnings.filterwarnings("ignore")

class MultiLevelSummarizer:
    """
    Handles three distinct levels of summarization for a comprehensive analysis.
    Uses a hybrid-model approach for the best quality at each level.
    """
    def __init__(self):
        """
        Initializes the Summarizer by loading two different models:
        1. BART for high-quality single-comment summaries.
        2. Flan-T5 for instruction-based thematic and executive summaries.
        """
        print("Initializing Multi-Level Summarizer...")
        try:
            # Model for Level 1: High-quality single comment summaries
            print(" - Loading Summarization model (BART-Large)...")
            self.l1_summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=0 # Use GPU if available, otherwise change to -1 for CPU
            )

            # Model for Levels 2 & 3: Instruction-based, structured summaries
            print(" - Loading Instruction-based LLM (Flan-T5-Large)...")
            self.l2_l3_summarizer = pipeline(
                "text2text-generation",
                model="google/flan-t5-large",
                device=0 # Use GPU if available, otherwise change to -1 for CPU
            )
            print("Multi-Level Summarizer initialized successfully.")
        except Exception as e:
            print(f"FATAL: Error initializing models: {e}")
            print("This may be a memory issue. Consider using smaller models or a machine with more VRAM/RAM.")
            self.l1_summarizer = None
            self.l2_l3_summarizer = None

    def generate_per_comment_summaries(self, comments_df: pd.DataFrame) -> pd.DataFrame:
        """LEVEL 1: Generates a summary for each individual comment."""
        if not self.l1_summarizer: return comments_df

        print("\n--- Generating Level 1: Per-Comment Summaries ---")
        summaries = []
        for text in comments_df['CommentText']:
            # Truncate text to be safe with model limits
            truncated_text = ' '.join(str(text).split()[:512])
            summary = self.l1_summarizer(truncated_text, max_length=60, min_length=15, do_sample=False)[0]['summary_text']
            summaries.append(summary)
        
        comments_df['L1_Summary'] = summaries
        return comments_df

    def generate_per_aspect_summaries(self, comments_df: pd.DataFrame) -> dict:
        """LEVEL 2: Generates a thematic summary for each unique SectionID."""
        if not self.l2_l3_summarizer: return {}

        print("\n--- Generating Level 2: Per-Aspect Thematic Summaries ---")
        aspect_summaries = {}
        # Group comments by the section they are about
        grouped = comments_df.groupby('SectionID')

        for aspect, group in grouped:
            if aspect == "Document Level": continue # Skip the general category for this level

            print(f" - Summarizing aspect: {aspect} ({len(group)} comments)...")
            # Combine all comments for this aspect into one large text block
            combined_text = "\n".join(group['CommentText'].tolist())
            
            # Create a detailed prompt for the instruction-tuned model
            prompt = (
                "Read the following public comments regarding the legal clause '{aspect}'. "
                "Synthesize the dominant arguments and concerns into a concise paragraph. "
                "If there are distinct positive and negative points, present them clearly. "
                "Conclude with the most common suggestion, if any.\n\n"
                "COMMENTS:\n{text}"
            ).format(aspect=aspect, text=combined_text[:3000]) # Truncate combined text

            summary = self.l2_l3_summarizer(prompt, max_length=200, num_beams=4, early_stopping=True)[0]['generated_text']
            aspect_summaries[aspect] = summary
            
        return aspect_summaries

    def generate_executive_summary(self, comments_df: pd.DataFrame) -> str:
        """LEVEL 3: Generates a single, high-level executive summary for all comments."""
        if not self.l2_l3_summarizer: return ""
        
        print("\n--- Generating Level 3: Overall Executive Summary ---")
        
        # Combine a sample of comments to create the input text
        # Using all 3000 would be too slow/long; a representative sample is better.
        sample_text = "\n".join(comments_df.sample(50)['CommentText'].tolist())
        
        # Create a high-level prompt
        prompt = (
            "You are a policy analyst assistant. Read the following sample of public comments on a draft government notification. "
            "Generate a one-paragraph executive summary that covers: "
            "1. The overall public sentiment. "
            "2. The most common positive arguments. "
            "3. The most critical concerns or negative arguments. "
            "4. The top recommendation or suggestion proposed by the public.\n\n"
            "COMMENTS:\n{text}"
        ).format(text=sample_text[:3000])

        summary = self.l2_l3_summarizer(prompt, max_length=256, num_beams=4, early_stopping=True)[0]['generated_text']
        return summary

# --- Main script to demonstrate the full workflow ---
if __name__ == '__main__':
    # Load your complete dataset
    try:
        df = pd.read_csv("C:\\Users\\Vansh\\Desktop\SIH\\Project\\dummy.csv")
        print(f"Loaded {len(df)} comments from 'dummy_mca_comments.csv'.")
    except FileNotFoundError:
        print("ERROR: 'dummy_mca_comments.csv' not found. Please generate it first.")
        exit()

    # Initialize the summarizer. This will load both models.
    multi_summarizer = MultiLevelSummarizer()

    # --- Run Level 1 ---
    df_with_summaries = multi_summarizer.generate_per_comment_summaries(df.head(5)) # Run on a sample of 5
    print("\n--- Level 1 Results (Sample) ---")
    for index, row in df_with_summaries.iterrows():
        print(f"CommentID: {row['CommentID']}")
        print(f"  Original: {row['CommentText'][:100]}...")
        print(f"  Summary: {row['L1_Summary']}\n")

    # --- Run Level 2 ---
    aspect_summaries_result = multi_summarizer.generate_per_aspect_summaries(df)
    print("\n--- Level 2 Results ---")
    for aspect, summary in aspect_summaries_result.items():
        print(f"Thematic Summary for Aspect '{aspect}':")
        print(f"  -> {summary}\n")

    # --- Run Level 3 ---
    executive_summary_result = multi_summarizer.generate_executive_summary(df)
    print("\n--- Level 3 Result ---")
    print("Overall Executive Summary:")
    print(f"-> {executive_summary_result}")