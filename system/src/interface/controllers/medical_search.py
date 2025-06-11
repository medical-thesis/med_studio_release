import os
import pandas as pd

data_dir = "system/src/core/medical_information"

complications_data = data_dir + "/complications.csv"
concepts_data = data_dir + "/concepts.csv"
symptoms_data = data_dir + "/symptoms.csv"
treatments_data = data_dir + "/treatments.csv"

df_symptoms = pd.read_csv(symptoms_data, encoding="utf-8")
df_treatments = pd.read_csv(treatments_data, encoding="utf-8")
df_complications = pd.read_csv(complications_data, encoding="utf-8")
df_concepts = pd.read_csv(concepts_data, encoding="utf-8")


dataset = pd.read_csv("system/dataset/processed/medquad_qa_pairs.csv", encoding="utf-8")

class MedicalSearchController:
    def __init__(self):
        self.items_per_page = 18 
        self.categories = list(dict.fromkeys(
            df_symptoms["cate"].tolist() +
            df_treatments["cate"].tolist() +
            df_complications["cate"].tolist() +
            df_concepts["cate"].tolist()
        ))
        self.dataset = dataset
    
    def get_medical_list(self):
        return sorted(self.categories)

    def get_questions_for_disease(self, disease_name):
        questions = []

        if disease_name in df_concepts["cate"].values:
            q = df_concepts[df_concepts["cate"] == disease_name]["Concept"].values[0]
            questions.append(q)

        if disease_name in df_symptoms["cate"].values:
            q = df_symptoms[df_symptoms["cate"] == disease_name]["Symptom"].values[0]
            questions.append(q)

        if disease_name in df_treatments["cate"].values:
            q = df_treatments[df_treatments["cate"] == disease_name]["Treatment"].values[0]
            questions.append(q)

        if disease_name in df_complications["cate"].values:
            q = df_complications[df_complications["cate"] == disease_name]["Complication"].values[0]
            questions.append(q)

        return questions
                
        
    