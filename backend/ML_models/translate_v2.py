import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from langdetect import detect

# --- Initialize models ONCE in your AIManager ---
# Note: This is a large model (1B parameters), loading it will require significant RAM/VRAM.
# Using torch.float16 and device_map can help manage memory.
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indictrans2-en-indic-1B", trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-en-indic-1B", trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")


# A mapping from langdetect codes to IndicTrans2 codes
LANG_CODE_MAP = {
    'hi': 'hin_Deva', # Hindi
    'mr': 'mar_Deva', # Marathi
    'ta': 'tam_Taml', # Tamil
    'te': 'tel_Telu', # Telugu
    'bn': 'ben_Beng', # Bengali
    'gu': 'guj_Gujr', # Gujarati
    # Add other mappings as needed
}

def translate_with_indictrans2(comment_text: str) -> str:
    """
    Detects Indian languages and translates them to English using IndicTrans2.
    """
    try:
        lang = detect(comment_text)

        if lang in LANG_CODE_MAP:
            src_lang = LANG_CODE_MAP[lang]
            tgt_lang = 'eng_Latn'
            
            # Add the language tokens to the comment text
            input_text = f"<{src_lang}> {comment_text} <{tgt_lang}>"
            
            print(f"Detected {lang}. Translating with IndicTrans2...")
            
            # Tokenize and translate
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
            generated_tokens = model.generate(**inputs, max_length=256, num_beams=5, num_return_sequences=1)
            
            # Decode the translated text
            translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
            return translated_text

        else: # Either English or an unsupported language
            return comment_text

    except Exception as e:
        print(f"Error during IndicTrans2 translation: {e}")
        return comment_text

# --- Example Usage ---
hindi_comment = "सरकार को छोटे व्यवसायों पर पड़ने वाले प्रभाव पर विचार करना चाहिए।"
marathi_comment = "या प्रस्तावाला माझा पूर्ण पाठिंबा आहे." # "I fully support this proposal."

translated_hindi = translate_with_indictrans2(hindi_comment)
translated_marathi = translate_with_indictrans2(marathi_comment)

print("-" * 20)
print(f"Original Hindi: {hindi_comment}")
print(f"Translated: {translated_hindi}")
print("-" * 20)
print(f"Original Marathi: {marathi_comment}")
print(f"Translated: {translated_marathi}")