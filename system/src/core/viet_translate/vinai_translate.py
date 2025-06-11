from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Literal


class VinaiTranslator():

    def __init__(
            self,
    ): pass

    def translate_vi2en(self, vi_text: str) -> str:
        tokenizer_vi2en = AutoTokenizer.from_pretrained("vinai/vinai-translate-vi2en-v2", src_lang="vi_VN")
        model_vi2en = AutoModelForSeq2SeqLM.from_pretrained("vinai/vinai-translate-vi2en-v2")

        input_ids = tokenizer_vi2en(vi_text, return_tensors="pt").input_ids
        output_ids = model_vi2en.generate(
            input_ids,
            decoder_start_token_id=tokenizer_vi2en.lang_code_to_id["en_XX"],
            num_return_sequences=1,
            num_beams=5,
            early_stopping=True
        )
        en_text = tokenizer_vi2en.batch_decode(output_ids, skip_special_tokens=True)
        en_text = " ".join(en_text)
        return en_text

    def translate_en2vi(self, en_text: str) -> str:
        tokenizer_en2vi = AutoTokenizer.from_pretrained("vinai/vinai-translate-en2vi-v2", src_lang="en_XX")
        model_en2vi = AutoModelForSeq2SeqLM.from_pretrained("vinai/vinai-translate-en2vi-v2")
        input_ids = tokenizer_en2vi(en_text, return_tensors="pt").input_ids
        output_ids = model_en2vi.generate(
            input_ids,
            decoder_start_token_id=tokenizer_en2vi.lang_code_to_id["vi_VN"],
            num_return_sequences=1,
            num_beams=5,
            early_stopping=True
        )
        vi_text = tokenizer_en2vi.batch_decode(output_ids, skip_special_tokens=True)
        vi_text = " ".join(vi_text)
        return vi_text


if __name__ == "__main__":
    print("\n\n\n ========= Example translate with VinAI: ======= \n\n")

    vi_text = "cô cho biết trước giờ tôi không đến phòng tập công cộng mà tập cùng giáo viên yoga riêng hoặc tự tập ở nhà khi tập thể dục trong không gian riêng tư tôi thoải mái dễ chịu hơn"
    print(VinaiTranslator().translate_vi2en(vi_text))

    en_text = "stroke, i haven't been to a public gym before, when i exercise in a private space i feel more comfortable"
    print(VinaiTranslator().translate_en2vi(en_text))
