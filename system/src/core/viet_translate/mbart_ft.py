from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class MBartFT():
    def __init__(self):
        pass

    def translate(self, text: str = "Kết luận: Viêm tai ứ dịch trên viêm V.a là bệnh lý hay gặp ở lứa tuổi trẻ em."):
        tokenizer = AutoTokenizer.from_pretrained(
            "facebook/mbart-large-50-many-to-many-mmt")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            pretrained_model_name_or_path="E:/source_code/nlp/med_studio_core/src/core/machine_translation/fine_tuning/content/out/ft")

        tokenizer.src_lang = "vi-VN"
        tokenizer.tgt_lang = "en-XX"
        encoded_vi = tokenizer(text, return_tensors="pt")
        generated_tokens = model.generate(
            **encoded_vi,
            forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
        )
        return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)


if __name__ == "__main__":
    text = "Ứng dụng một số phần mềm phân tích mã vạch DNA để định danh cây dược liệu"
    translated = MBartFT().translate(text=text)
    print("translated: ", translated)
