from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os
import sys
from dotenv import load_dotenv
load_dotenv(dotenv_path="system/src/core/config/.env")

project_root = os.getenv("PROJECT_ROOT")

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from system.src.core.components.api.api_keys import APIKeyManager
from system.src.core.components.query_process.query_translation import QueryTranslator, QueryTranslator_GEMINI

translator = QueryTranslator()
translator_gemini = QueryTranslator_GEMINI(
    model="gemini-1.5-pro",
    api_key=APIKeyManager("GEMINI_API_KEY"),
)

from system.src.core.viet_translate.opus_translate import OpusTranslator
opus_translate = OpusTranslator()

from system.src.core.viet_translate.vietai_envit5 import VietaiEnviT5Translator
translator_vietai = VietaiEnviT5Translator()

from system.src.core.viet_translate.vinai_translate import VinaiTranslator
translator_vinai = VinaiTranslator()  

class TranslateController:
    def __init__(self):
        pass

    def translate(self, text, source_lang, target_lang, model=["cohere", "opus", "vietai", "vinai"]):
        
        if model == "cohere":
            if source_lang == "en" and target_lang == "vi":
                return translator.translate_english_to_vietnamese(text=text)
            elif source_lang == "vi" and target_lang == "en":
                return translator.translate_vietnamese_to_english(text=text)
        
        if model == "opus":
            if source_lang == "en" and target_lang == "vi":
                return opus_translate.translate(src_lang="en", target_lang="vi", text=text)
            elif source_lang == "vi" and target_lang == "en":
                return opus_translate.translate(src_lang="vi", target_lang="en", text=text)
        
        if model == "vietai":
            if source_lang == "en" and target_lang == "vi":
                return translator_vietai.translate(text=text)
            elif source_lang == "vi" and target_lang == "en":
                return translator_vietai.translate(text=text)
        
        if model == "vinai":
            if source_lang == "en" and target_lang == "vi":
                return translator_vinai.translate_en2vi(text)
            elif source_lang == "vi" and target_lang == "en":
                return translator_vinai.translate_vi2en(text)