import streamlit as st
from app import evaluete_scores_scarebleu, evaluete_scores_bluert, evaluete_scores_comet
import pandas as pd
import os



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
test_out_file = 'data/wmt22/cs-uk/test.uk'
with open(test_out_file,"r") as t:
    test_out_file_is = t.readlines()
    st.success(f"Loaded{test_out_file}automatically")
    #uploaded_source = st.file_uploader("Upload the source file (TXT, optional for COMET)", type=["txt"], key="source")
train_out_file = 'data/wmt22/cs-uk/train.uk.30k'
with open(train_out_file,"r") as t:
    train_out_file_is = t.readlines()
    st.success(f"Loaded{train_out_file}automatically")



#filter for score methods to display
score_method = ['sacreBLUE','Bleurt','Comet']
selected_score_method = st.selectbox("Select a score method to display:",score_method)



source_for_trainning = "data/wmt22/cs-uk/train.cs.30k"
with open(source_for_trainning,"r") as l:
    source_for_trainning_is = l.readlines()
    st.success(f"Loaded{source_for_trainning}automatically")

source_for_testing = "data/wmt22/cs-uk/test.cs"
with open(source_for_testing,"r") as p:
    source_for_testing_is = p.readlines()
    st.success(f"Loaded{source_for_testing}automatically")
    



#logic
if selected_model_label:
    if uploaded_files_training and uploaded_files_testing and test_out_file_is and train_out_file_is and source_for_trainning_is and source_for_testing_is:
        st.write("Ready for scareBLEU and BLEURT scoring!")
        if st.button("Generate Matching Table and Evaluate Scores"):
            if selected_score_method == "sacreBLUE":
                try:
                    for name_of_the_file,needed_file in uploaded_files_testing.items():
                        result = evaluete_scores_scarebleu(needed_file, test_out_file_is,f"{selected_model_label}_{name_of_the_file}")
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
                        result1 = evaluete_scores_scarebleu(needed_file1, train_out_file_is,f"{selected_model_label}_{name_of_the_file1}")
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
                        result = evaluete_scores_bluert(needed_file,test_out_file_is,f"{selected_model_label}_{name_of_the_file}")
                        avg_bleurt,score_value = result

                    
                        matched_sentences = list(zip(needed_file, test_out_file_is,score_value))
                        df = pd.DataFrame(matched_sentences, columns=["Machine-Generated Sentences", "Standard Translated Sentences", "Bleurt Score"])
                        df["Bleurt Score"] =df["Bleurt Score"].astype(str)
                        df.insert(0, "File Name", name_of_the_file)
                        avg_row = {
                                     "Machine-Generated Sentences": "AVERAGE",
                                     "Standard Translated Sentences": "AVERAGE",
                                     "Bleurt Average Score": str(avg_bleurt),
                                    }
                        df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
                        st.dataframe(df)

                    for name_of_the_file1,needed_file1 in uploaded_files_training.items():
                        result1 = evaluete_scores_bluert(needed_file1, train_out_file_is,f"{selected_model_label}_{name_of_the_file1}")
                        avg_bleurt1,score_value1 = result1

                        matched_sentences1 = list(zip(needed_file1, train_out_file_is,score_value1))
                        df1 = pd.DataFrame(matched_sentences1, columns=["Machine-Generated Sentences", "Standard Translated Sentences", "bleurt Score"])
                        df["Bleurt Score"] = df["Bleurt Score"].astype(str)
                        df1.insert(0, "File Name", name_of_the_file1)
                        avg_row1 = {
                            "Machine-Generated Sentences": "AVERAGE",
                            "Standard Translated Sentences": "AVERAGE",
                            "Bleurt Average Score": str(avg_bleurt1),
                                    }
                        df1 = pd.concat([df1, pd.DataFrame([avg_row1])], ignore_index=True)
                        st.dataframe(df1)

                except Exception as e:
                    st.error(f"An error occurred: {e}")
            if selected_score_method == "Comet":
                try:
                    for name_of_the_file,needed_file in uploaded_files_testing.items():
                        print(f'working on file {name_of_the_file} in first for loop')
                        result = evaluete_scores_comet(needed_file, test_out_file_is,source_for_testing_is,f"{selected_model_label}_{name_of_the_file}")
                        comet_scores, avg_comet= result


                        matched_sentences = list(zip(needed_file, test_out_file_is,comet_scores))
                        df = pd.DataFrame(matched_sentences, columns=["Machine-Generated Sentences", "Standard Translated Sentences", "Comet Score"])
                        df["Comet Score"] =df["Comet Score"].astype(str)
                        df.insert(0, "File Name", name_of_the_file)
                        avg_row = {
                            "Machine-Generated Sentences": "AVERAGE",
                            "Standard Translated Sentences": "AVERAGE",
                            "Comet Average Score": str(avg_comet),
                                    }
                        df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
                        st.dataframe(df)

                    for name_of_the_file1,needed_file1 in uploaded_files_training.items():
                        print(f'working on file {name_of_the_file} in second for loop')

                        result1 = evaluete_scores_comet(needed_file1, train_out_file_is,source_for_trainning_is,f"{selected_model_label}_{name_of_the_file1}")
                        comet_scores1, avg_comet1 = result1

                        matched_sentences1 = list(zip(needed_file1, train_out_file_is,comet_scores1))
                        df1 = pd.DataFrame(matched_sentences1, columns=["Machine-Generated Sentences", "Standard Translated Sentences", "Comet Score"])
                        df["Comet Score"] =df["Comet Score"].astype(str)
                        df1.insert(0, "File Name", name_of_the_file1)
                        avg_row1 = {
                            "Machine-Generated Sentences": "AVERAGE",
                            "Standard Translated Sentences": "AVERAGE",
                            "Comet Average Score": str(avg_comet1),
                                    }
                        df1 = pd.concat([df1, pd.DataFrame([avg_row1])], ignore_index=True)
                        st.dataframe(df1)

                except Exception as e:
                    st.error(f"An error occurred: {e}")


    # if uploaded_files_training and uploaded_files_testing and test_out_file_is and train_out_file_is and source_for_trainning_is and source_for_testing_is:
    #     if selected_score_method == "Comet":
    #         if st.button("Generate Matching Table and Evaluate Scores"):
    #             print('calculating comet')
    #             try:
    #                 for name_of_the_file,needed_file in uploaded_files_testing.items():
    #                     print(f'working on file {name_of_the_file} in first for loop')
    #                     result = evaluete_scores_comet(needed_file, test_out_file_is,source_for_testing_is)
    #                     comet_scores, avg_comet= result


    #                     matched_sentences = list(zip(needed_file, test_out_file_is,comet_scores))
    #                     df = pd.DataFrame(matched_sentences, columns=["Machine-Generated Sentences", "Standard Translated Sentences", "Comet Score"])
    #                     df["Comet Score"] =df["Comet Score"].astype(str)
    #                     df.insert(0, "File Name", name_of_the_file)
    #                     avg_row = {
    #                         "Machine-Generated Sentences": "AVERAGE",
    #                         "Standard Translated Sentences": "AVERAGE",
    #                         "Comet Average Score": str(avg_comet),
    #                                 }
    #                     df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
    #                     st.dataframe(df)

    #                 for name_of_the_file1,needed_file1 in uploaded_files_training.items():
    #                     print(f'working on file {name_of_the_file} in second for loop')

    #                     result1 = evaluete_scores_bluert(needed_file1, train_out_file_is,source_for_trainning_is)
    #                     comet_scores1, avg_comet1 = result1

    #                     matched_sentences1 = list(zip(needed_file1, train_out_file_is,comet_scores1))
    #                     df1 = pd.DataFrame(matched_sentences1, columns=["Machine-Generated Sentences", "Standard Translated Sentences", "Comet Score"])
    #                     df["Comet Score"] =df["Comet Score"].astype(str)
    #                     df1.insert(0, "File Name", name_of_the_file1)
    #                     avg_row1 = {
    #                         "Machine-Generated Sentences": "AVERAGE",
    #                         "Standard Translated Sentences": "AVERAGE",
    #                         "Comet Average Score": str(avg_comet1),
    #                                 }
    #                     df1 = pd.concat([df1, pd.DataFrame([avg_row1])], ignore_index=True)
    #                     st.dataframe(df1)

    #             except Exception as e:
    #                 st.error(f"An error occurred: {e}")


