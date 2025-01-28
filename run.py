import streamlit as st
from app import extract_lines_from_txt, split_into_sentences,evaluate_scores
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
uploaded_files = {} 
if selected_model_label:
    st.subheader(f"Upload files for {selected_model_label}") 
    for file_index, file_name in enumerate(result_files_in_model[selected_model_label], start=1):
        file_path = os.path.join(results_folder,selected_model_label, file_name)
        uploaded_file = st.file_uploader(
            f"Upload the machine-translated file for {file_name}",
            type=["txt"],
            key=f"{selected_model_label}_{file_index}"
        )
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                uploaded_files[file_name] = f.read()
                st.success(f"Loaded {file_name} automatically.")
        if st.button("X",key=f"delete {file_name}"): 
            uploaded_files = uploaded_files.pop(uploaded_files[file_name])
            st.success(f"{file_name} deleted.")
            

    uploaded_file2 = st.file_uploader("Upload the human-generated (standard) file (TXT)", type=["txt"], key="file2")
    uploaded_source = st.file_uploader("Upload the source file (TXT, optional for COMET)", type=["txt"], key="source")



#filter for score methods to display
if uploaded_source:
    score_method = ['BLEURT','sacreBLUE','Comet']
else:
    score_method = ['BLEURT','sacreBLUE']
selected_score_method = st.selectbox("Select a score method to display:",score_method)


paragraphs_need_to_saperate ={}
#path to each file to sparate to sentence 
for name,content in uploaded_files.items():
                paragraphs = split_into_sentences([content])
                paragraphs_need_to_saperate[name] = paragraphs



#logic
if selected_model_label:
    if uploaded_files and uploaded_file2:
        st.write("Ready for BLEU scoring!")
        if uploaded_source:
            st.write("Source file uploaded for COMET evaluation.")
        if st.button("Generate Matching Table and Evaluate Scores"):
            try:
                # Extract lines from uploaded files
                lines2 = extract_lines_from_txt(uploaded_file2)
                source_lines = extract_lines_from_txt(uploaded_source) if uploaded_source else None

                # Split lines into sentences
                sentences2 = split_into_sentences(lines2)
                source_sentences = split_into_sentences(source_lines) if source_lines else None

                # Evaluate scores
                # Store results for all sentences
                for needed_file in paragraphs_need_to_saperate.values():
                    needed_file = list(needed_file)
                    result = evaluate_scores(needed_file, sentences2, source_sentences)
                    bleurt_scores, scarebleu_scores, comet_scores, avg_bleurt, avg_scarebleu, avg_comet, source_sentences_used = result  # Evaluate scores
                    if selected_score_method =="BLEURT":
                        matched_sentences = list(zip(needed_file, sentences2))
                        df = pd.DataFrame(matched_sentences, columns=["Machine-Generated Sentences", "Standard Translated Sentences"])
                        df["BLEURT Score"] = bleurt_scores
                        avg_row = {
                            "Machine-Generated Sentences": "AVERAGE",
                            "Standard Translated Sentences": "AVERAGE",
                            "BLEURT Score": avg_bleurt,
                                    }
                        df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
                        st.dataframe(df)
                    elif selected_score_method == "sacreBLUE":
                        matched_sentences = list(zip(needed_file, sentences2))
                        df = pd.DataFrame(matched_sentences, columns=["Machine-Generated Sentences", "Standard Translated Sentences"])
                        df["SacréBLEU Score"] = scarebleu_scores
                        avg_row = {
                            "Machine-Generated Sentences": "AVERAGE",
                            "Standard Translated Sentences": "AVERAGE",
                            "SacréBLEU Score": avg_scarebleu,
                                    }
                        df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
                        st.dataframe(df)
                    elif selected_score_method =="Comet" and uploaded_source:
                        matched_sentences = list(zip(source_sentences_used, needed_file, sentences2))
                        df = pd.DataFrame(matched_sentences, columns=["Source Sentences", "Machine-Generated Sentences", "Standard Translated Sentences"])
                        df["COMET Score"] = comet_scores
                        avg_row = {
                            "Source Sentences":"AVERAGE",
                            "Machine-Generated Sentences": "AVERAGE",
                            "Standard Translated Sentences": "AVERAGE",
                            "COMET Score": avg_comet,
                                    }
                        df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
                        st.dataframe(df)


    
            except Exception as e:
                st.error(f"An error occurred: {e}")







