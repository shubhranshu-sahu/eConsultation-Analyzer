import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from langdetect import detect

# --- Initialize models ONCE in your AIManager ---
# Note: This is a large model (1B parameters), loading it will require significant RAM/VRAM.
# Using torch.float16 and device_map can help manage memory.
# tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indictrans2-en-indic-1B", trust_remote_code=True)
# model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-en-indic-1B", trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor
from langdetect import detect

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device set to use {DEVICE}")

# Model and tokenizer initialization
model_name = "ai4bharat/indictrans2-en-indic-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    attn_implementation="flash_attention_2"
).to(DEVICE)

ip = IndicProcessor(inference=True)

# Mapping for langdetect to IndicTrans2 source language codes
LANG_CODE_MAP = {
    'hi': 'hin_Deva',  # Hindi
    'mr': 'mar_Deva',  # Marathi
    'ta': 'tam_Taml',  # Tamil
    'te': 'tel_Telu',  # Telugu
    'bn': 'ben_Beng',  # Bengali
    'gu': 'guj_Gujr',  # Gujarati
}

def translate_with_indictrans2(comment_text: str) -> str:
    try:
        lang = detect(comment_text)

        if lang in LANG_CODE_MAP:
            src_lang = LANG_CODE_MAP[lang]
            tgt_lang = "eng_Latn"
            
            print(f"Detected {lang}. Translating with IndicTrans2...")

            # Preprocess input
            batch = ip.preprocess_batch(
                [comment_text],
                src_lang=src_lang,
                tgt_lang=tgt_lang,
            )

            inputs = tokenizer(
                batch,
                truncation=True,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
            ).to(DEVICE)

            # Generate translation
            with torch.no_grad():
                generated_tokens = model.generate(
                    **inputs,
                    use_cache=True,
                    min_length=0,
                    max_length=256,
                    num_beams=5,
                    num_return_sequences=1,
                )

            generated_tokens = tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            # Postprocess output
            translations = ip.postprocess_batch(generated_tokens, lang=tgt_lang)

            return translations[0]

        else:
            return comment_text  # English or unsupported language

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