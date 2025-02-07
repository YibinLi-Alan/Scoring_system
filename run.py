import streamlit as st
from app import evaluete_scores_scarebleu, evaluete_scores_bluert
import pandas as pd
import os
# Streamlit app

st.title("TXT File Upload and Sentence Matching with Evaluation")

#sort and find out how many models we have and find out all models
results_folder = "results/wmt22/cs-uk"
models = [folder for folder in os.listdir(results_folder) if os.path.isdir(os.path.join(results_folder, folder))]
models_sorted = sorted(models, key=lambda x: int(x.split('_')[-1]))  # Extract the number and sort numerically
selected_model_label = st.selectbox("Select a model from the results folder:", models_sorted)

# find out all files for each model
result_files_in_model ={}

for model in models_sorted :
    which_model =os.path.join(results_folder,model)
    result_files_in_model[model] = [result_file for result_file in os.listdir(which_model)]


    
#create all space for electing machine tranlated files - in future maybe can automaticlly upload
uploaded_files_training = {}
uploaded_files_testing ={}
if selected_model_label:
    st.subheader(f"Upload files for {selected_model_label}") 
    for file_index, file_name in enumerate(result_files_in_model[selected_model_label], start=1):
        file_path = os.path.join(results_folder,selected_model_label, file_name)
        if "training" in file_name:
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    uploaded_files_training[file_name] = f.readlines()
                    st.success(f"Loaded {file_name} automatically.")
        else:
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    uploaded_files_testing[file_name] = f.readlines()
                    st.success(f"Loaded {file_name} automatically.")

#automaticlly upload test file  
test_out_file = 'data/wmt22/cs-uk/test.us.txt'
with open(test_out_file,"r") as t:
    test_out_file_is = t.readlines()
    st.success(f"Loaded{test_out_file}automatically")
    #uploaded_source = st.file_uploader("Upload the source file (TXT, optional for COMET)", type=["txt"], key="source")
train_out_file = 'data/wmt22/cs-uk/train.uk.30k'
with open(train_out_file,"r") as t:
    train_out_file_is = t.readlines()
    st.success(f"Loaded{train_out_file}automatically")



#filter for score methods to display
# if uploaded_source:
#     score_method = ['BLEURT','sacreBLUE','Comet']
# else:
#     score_method = ['BLEURT','sacreBLUE']
score_method = ['sacreBLUE','Bleurt']
selected_score_method = st.selectbox("Select a score method to display:",score_method)
#path to each file to sparate to sentence 


#logic
if selected_model_label:
    if uploaded_files_training and uploaded_files_testing and test_out_file_is and train_out_file_is:
        st.write("Ready for scareBLEU and BLEURT scoring!")
        if st.button("Generate Matching Table and Evaluate Scores"):
            if selected_score_method == "sacreBLUE":
                try:
                    for name_of_the_file,needed_file in uploaded_files_testing.items():
                        result = evaluete_scores_scarebleu(needed_file, test_out_file_is)
                        scarebleu_scores, avg_scarebleu = result

                    
                        matched_sentences = list(zip(needed_file, test_out_file_is))
                        df = pd.DataFrame(matched_sentences, columns=["Machine-Generated Sentences", "Standard Translated Sentences"])
                        df.insert(0, "File Name", name_of_the_file)
                        df["SacréBLEU Score"] = scarebleu_scores
                        avg_row = {
                            "Machine-Generated Sentences": "AVERAGE",
                            "Standard Translated Sentences": "AVERAGE",
                            "SacréBLEU Score": avg_scarebleu,
                                    }
                        df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
                        st.dataframe(df)

                    for name_of_the_file1,needed_file1 in uploaded_files_training.items():
                        result1 = evaluete_scores_scarebleu(needed_file1, train_out_file_is)
                        scarebleu_scores1, avg_scarebleu1 = result1

                        matched_sentences1 = list(zip(needed_file1, train_out_file_is))
                        df1 = pd.DataFrame(matched_sentences1, columns=["Machine-Generated Sentences", "Standard Translated Sentences"])
                        df1.insert(0, "File Name", name_of_the_file1)
                        df1["SacréBLEU Score"] = scarebleu_scores1
                        avg_row1 = {
                            "Machine-Generated Sentences": "AVERAGE",
                            "Standard Translated Sentences": "AVERAGE",
                            "SacréBLEU Score": avg_scarebleu1,
                                    }
                        df1 = pd.concat([df1, pd.DataFrame([avg_row1])], ignore_index=True)
                        st.dataframe(df1)
        
                except Exception as e:
                    st.error(f"An error occurred: {e}")

            if selected_score_method == "Bleurt":
                try:
                    for name_of_the_file,needed_file in uploaded_files_testing.items():
                        result = evaluete_scores_bluert(needed_file, test_out_file_is)
                        avg_bleurt,bleurt_scores = result

                    
                        matched_sentences = list(zip(needed_file, test_out_file_is))
                        df = pd.DataFrame(matched_sentences, columns=["Machine-Generated Sentences", "Standard Translated Sentences"])
                        df.insert(0, "File Name", name_of_the_file)
                        df["SacréBLEU Score"] = bleurt_scores
                        avg_row = {
                            "Machine-Generated Sentences": "AVERAGE",
                            "Standard Translated Sentences": "AVERAGE",
                            "SacréBLEU Score": avg_bleurt,
                                    }
                        df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
                        st.dataframe(df)

                    for name_of_the_file1,needed_file1 in uploaded_files_training.items():
                        result1 = evaluete_scores_bluert(needed_file1, train_out_file_is)
                        avg_bleurt1,bleurt_scores1 = result1

                        matched_sentences1 = list(zip(needed_file1, train_out_file_is))
                        df1 = pd.DataFrame(matched_sentences1, columns=["Machine-Generated Sentences", "Standard Translated Sentences"])
                        df1.insert(0, "File Name", name_of_the_file1)
                        df1["SacréBLEU Score"] = bleurt_scores1
                        avg_row1 = {
                            "Machine-Generated Sentences": "AVERAGE",
                            "Standard Translated Sentences": "AVERAGE",
                            "SacréBLEU Score": avg_bleurt1,
                                    }
                        df1 = pd.concat([df1, pd.DataFrame([avg_row1])], ignore_index=True)
                        st.dataframe(df1)

                except Exception as e:
                    st.error(f"An error occurred: {e}") 







