"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# 1. Define the model you want to download and the local path to save it to
MODEL_NAME = "ProsusAI/finbert"
SAVE_DIRECTORY = "./finbert-local"

def download_and_save_model():
    
    # Downloads the FinBERT model and tokenizer from Hugging Face  and saves them to a local directory.
    
    # Create the directory if it doesn't exist
    if not os.path.exists(SAVE_DIRECTORY):
        print(f"Creating directory: {SAVE_DIRECTORY}")
        os.makedirs(SAVE_DIRECTORY)

    # Check if the main model file already exists to avoid re-downloading
    if os.path.exists(os.path.join(SAVE_DIRECTORY, "pytorch_model.bin")):
        print(f"Model files already exist in {SAVE_DIRECTORY}. Skipping download.")
        print("If you want to re-download, please delete the 'finbert-local' folder first.")
        return

    print(f"Downloading model and tokenizer for '{MODEL_NAME}'...")
    
    try:
        # Download the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # Download the model
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

        # Save the tokenizer and model to the specified directory
        tokenizer.save_pretrained(SAVE_DIRECTORY)
        model.save_pretrained(SAVE_DIRECTORY)
        
        print(f"Model and tokenizer saved successfully to '{SAVE_DIRECTORY}'")

    except Exception as e:
        print(f"An error occurred during download: {e}")
        print("Please check your internet connection and the model name.")


if __name__ == "__main__":
    download_and_save_model()

"""