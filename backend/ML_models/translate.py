from transformers import pipeline
from langdetect import detect

# This would be loaded once in your AIManager's __init__ method
translator_pipeline = pipeline("translation", model="Helsinki-NLP/opus-mt-mul-en")

def translate_if_needed(comment_text: str) -> str:
    """
    Detects the language of a comment and translates it to English if necessary.
    """
    try:
        # Detect the language of the comment
        lang = detect(comment_text)

        # If the language is not English, translate it
        if lang != 'en':
            print(f"Detected non-English comment (lang: {lang}). Translating...")
            translated_result = translator_pipeline(comment_text)
            return translated_result[0]['translation_text']
        else:
            # If it's already English, return it as is
            return comment_text
    except Exception as e:
        print(f"Error during language detection or translation: {e}")
        # Fallback to returning the original text if an error occurs
        return comment_text

# --- Example Usage ---
hindi_comment = "मैं इस विषय को लेकर बहुत चिंतित हूँ" # "The government should consider the impact on small businesses."
english_comment = "This is a good proposal."

translated_hindi = translate_if_needed(hindi_comment)
translated_english = translate_if_needed(english_comment)

print(f"Original Hindi: {hindi_comment}")
print(f"Translated: {translated_hindi}")
print("-" * 20)
print(f"Original English: {english_comment}")
print(f"Translated: {translated_english}")