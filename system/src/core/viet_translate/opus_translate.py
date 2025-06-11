from transformers import pipeline

class OpusTranslator():
    def __init__(self):
        self.pipe = None

    def translate(self, src_lang: str = "en", target_lang: str = "vi", text: str = ""):
        if src_lang == "en" and target_lang == "vi":
            self.pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-en-vi")
        elif src_lang == "vi" and target_lang == "en":
            self.pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-vi-en")

        
        return self.pipe(text)[0]["translation_text"]
    

if __name__ == "__main__":
    trans = OpusTranslator().translate(src_lang="vi", target_lang="en", text="bạn có thể giải thích cho tôi về bệnh đột quỵ không ?")
    print("translated: ", trans)