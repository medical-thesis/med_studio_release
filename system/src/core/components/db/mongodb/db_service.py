from interaction import MongoDBManager
import pandas as pd
from connection import MongoDBConnection
from datetime import datetime

uri = 'mongodb+srv://trieu:trieu_ute@medstudiocluster-v1.wkjpnqc.mongodb.net/?retryWrites=true&w=majority&appName=MedStudioCluster-v1'
connector = MongoDBConnection(uri, db_name='med_studio_dev1')

connector.connect()
current_db = connector.get_db()

df = pd.read_csv('../../../dataset/processed/medquad_qa_pairs.csv')
print("df.describe: ", df.describe())
print("df.head: ", df.head())

unique_source = df['source'].unique().tolist()
print("unique_source: ", unique_source)

clts_name = ['senior_health',
             'comprehensive_cancer',
             'medline_plus_health_topics',
             'genetic_and_rare_diseases',
             'disease_control_and_prevention',
             'heart_lung_and_blood',
             'neurological_disorders_and_stroke',
             'growth_hormone_receptor',
             'diabetes_and_digestive_and_kidney_diseases']

print("clts_name: ", clts_name)
print(len(clts_name))

dbmanager = MongoDBManager(db=current_db)

clts = {}
for i in clts_name:
    print('colection: ', i)
    current_clt = dbmanager.create_collection(collection_name=i)
    clts[i] = current_clt
    print()

print("\n\n ============== Started, ...")
try:
    for _, row in df.iterrows():

        data = {}

        data = {
            "question": str(row['question']),
            "answer": str(row['answer']),
            "metadata": {
                "source_dataset": 'MedQuAD',
                "source_question": str(row['source']),
                "focus_area": str(row['focus_area'])
            }
        }

        if data:
            if row['source'] == 'Senior Health':
                dbmanager.set_collection(clts['senior_health'])
                dbmanager.insert_data(data=data)
                current_time = datetime.now().isoformat(timespec='seconds')
                print(f"[{current_time}] Insert success 'Senior Health' ...! \n")
            elif row['source'] == 'Comprehensive Cancer':
                dbmanager.set_collection(clts['comprehensive_cancer'])
                dbmanager.insert_data(data=data)
                current_time = datetime.now().isoformat(timespec='seconds')
                print(
                    f"[{current_time}] Insert success 'Comprehensive Cancer' ...! \n")
            elif row['source'] == 'Medline Plus, Health Topics':
                dbmanager.set_collection(clts['medline_plus_health_topics'])
                dbmanager.insert_data(data=data)
                current_time = datetime.now().isoformat(timespec='seconds')
                print(
                    f"[{current_time}] Insert success 'Medline Plus, Health Topics' ...! \n")
            elif row['source'] == 'Genetic and Rare Diseases (GARD)':
                dbmanager.set_collection(clts['genetic_and_rare_diseases'])
                dbmanager.insert_data(data=data)
                current_time = datetime.now().isoformat(timespec='seconds')
                print(
                    f"[{current_time}] Insert success 'Genetic and Rare Diseases (GARD)' ...! \n")
            elif row['source'] == 'Disease Control and Prevention (CDC)':
                dbmanager.set_collection(
                    clts['disease_control_and_prevention'])
                dbmanager.insert_data(data=data)
                current_time = datetime.now().isoformat(timespec='seconds')
                print(
                    f"[{current_time}] Insert success 'Disease Control and Prevention (CDC)' ...! \n")
            elif row['source'] == 'Heart, Lung, and Blood (NHLBI)':
                dbmanager.set_collection(clts['heart_lung_and_blood'])
                dbmanager.insert_data(data=data)
                current_time = datetime.now().isoformat(timespec='seconds')
                print(
                    f"[{current_time}] Insert success 'Heart, Lung, and Blood (NHLBI)' ...! \n")
            elif row['source'] == 'Neurological Disorders and Stroke (NINDS)':
                dbmanager.set_collection(
                    clts['neurological_disorders_and_stroke'])
                dbmanager.insert_data(data=data)
                current_time = datetime.now().isoformat(timespec='seconds')
                print(
                    f"[{current_time}] Insert success 'Neurological Disorders and Stroke (NINDS)' ...! \n")
            elif row['source'] == 'Growth Hormone Receptor (GHR)':
                dbmanager.set_collection(clts['growth_hormone_receptor'])
                dbmanager.insert_data(data=data)
                current_time = datetime.now().isoformat(timespec='seconds')
                print(
                    f"[{current_time}] Insert success 'Growth Hormone Receptor (GHR)' ...! \n")
            elif row['source'] == 'Diabetes and Digestive and Kidney Diseases (NIDDK)':
                dbmanager.set_collection(
                    clts['diabetes_and_digestive_and_kidney_diseases'])
                dbmanager.insert_data(data=data)
                current_time = datetime.now().isoformat(timespec='seconds')
                print(
                    f"[{current_time}] Insert success 'Diabetes and Digestive and Kidney Diseases (NIDDK)' ...! \n\n")
except Exception as e:
    print(e)

print("\nCompleted, ...")
