from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Literal, List

class VietaiEnviT5Translator():
    def __init__(self):
        self.model_name = "VietAI/envit5-translation"
        self.tokenizer = None
        self.model = None
        self.set_model()

    def set_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.model_name
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            pretrained_model_name_or_path=self.model_name
        )

    def translate(self, text: str) -> str:
        input_ids = self.tokenizer(
            text, 
            return_tensors="pt"
        ).input_ids
        output_ids = self.model.generate(
            input_ids, 
            max_length=512
        )
        translated_text = self.tokenizer.batch_decode(
            output_ids, 
            skip_special_tokens=True
        )

        translated_text = " ".join(translated_text)
        return translated_text

    def translate_batch(self, inputs: List[str]) -> List[str]:
        input_ids = self.tokenizer(
            inputs,
            return_tensors="pt",
            padding=True
        ).input_ids
        output_ids = self.model.generate(
            input_ids,
            max_length=512
        )
        translated_text = self.tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True
        )

        return translated_text

    def translate_with_prefix(self, text: str, src_lang: Literal["en", "vi"]) -> str:
        if src_lang == "en": text = "en: " + text
        elif src_lang == "vi": text = "vi: " + text

        input_ids = self.tokenizer(
            text, 
            return_tensors="pt"
        ).input_ids
        output_ids = self.model.generate(
            input_ids, 
            max_length=512
        )
        translated_text = self.tokenizer.batch_decode(
            output_ids, 
            skip_special_tokens=True
        )

        translated_text = " ".join(translated_text)
        return translated_text

if __name__ == "__main__":
    print("\n\n\n ========= Example translate with VietAI: ======= \n\n")
    vi_text = "ủa sao tôi uống thuốc rồi mà vẫn thấy đau đầu nhỉ."
    print(VietaiEnviT5Translator().translate(text=vi_text))

    en_text = "hello, how are you today ?"
    print(VietaiEnviT5Translator().translate(text=en_text))
