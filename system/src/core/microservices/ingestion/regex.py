import re
import pandas as pd

if __name__ == "__main__":
    dt = pd.read_csv("system/core/dataset/raw/prepared_generated_data_for_nhs_uk_qa.csv")

    new_dt = pd.DataFrame(columns=["question", "answer", "source", "category","link"])
    text = dt["text"].values
    for i in text:
        regex = r"<\|user\|>\s*((?:.|\n)*?)\s*<\|eos\|>\s*<\|ai\|>\s*((?:.|\n)*?)\s*References:\s*((?:.|\n)*?)\s*<\|eos\|>\s*<\|eod\|>"
        matches = re.findall(regex, i)
        for match in matches:
            question = match[0].strip()
            answer = match[1].strip()
            references = match[2].strip()
            link = references.lstrip("- ").strip()
            
            category = re.findall(r"/conditions/([^/]+)/", link)
            category = " ".join(category)
            source = "NHS"
            new_dt = new_dt._append({"question": question, "answer": answer, "source": source, "category": category,"link": link}, ignore_index=True)
    new_dt.to_csv("system/core/dataset/processed/processed_data_for_nhs_uk_qa.csv", index=False)
    
    dt = pd.read_csv("system/core/dataset/raw/prepared_generated_data_for_medical_tasks.csv")
    new_dt = pd.DataFrame(columns=["question", "answer"])
    text = dt["text"].values
    for i in text:
        regex = r"<\|user\|>\s*((?:.|\n)*?)\s*<\|eos\|>\s*<\|ai\|>\s*((?:.|\n)*?)\s*<\|eos\|>\s*<\|eod\|>"
        matches = re.findall(regex, i)
        for match in matches:
            question = match[0].strip()
            answer = match[1].strip()
            new_dt = new_dt._append({"question": question, "answer": answer}, ignore_index=True)
    new_dt.to_csv("system/core/dataset/processed/processed_data_for_medical_tasks.csv", index=False)
    
    dt = pd.read_csv("system/core/dataset/raw/MedQuAD.csv")
    new_dt = pd.DataFrame(columns=["question", "answer", "link"])
    text = dt["Answer"].values
    for i in text:
        regex = r"\s*Question:\s*((?:.|\n)*?)\s*URL:\s*((?:.|\n)*?)\s*Answer:\s*(.*)"
        matches = re.findall(regex, i, re.DOTALL)
        for match in matches:
            question = match[0].strip()
            answer = match[2].strip()
            link = match[1].strip()
            new_dt = new_dt._append({"question": question, "answer": answer, "link": link}, ignore_index=True)
    new_dt.to_csv("system/core/dataset/processed/processed_data_for_MedQuAD.csv", index=False)