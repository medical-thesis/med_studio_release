from deep_translator import GoogleTranslator
import pandas as pd

raw_data = "E:/source_code/nlp/med_studio/system/core/dataset/raw/medical_tc_train.csv"
processed_data_path = "E:/source_code/nlp/med_studio/system/core/dataset/processed/medical_tc_train.csv"

def translate(text: str, src: str = "en", dest: str = "vi"):
    return GoogleTranslator(source=src, target=dest).translate(text=text)

df = pd.read_csv(raw_data)

processed_data = []
for index, row in df.iterrows():
    print("index: ", index)
    item = [
        row["condition_label"],
        row["medical_abstract"],
        translate(text=row["medical_abstract"])
    ]
    processed_data.append(item)
processed_df = pd.DataFrame(processed_data, columns=["condition_label", "medical_abstract_origin", "medical_abstract_translated"])
processed_df.to_csv(processed_data_path)
print("\n\n========= Data after processed: =========")
