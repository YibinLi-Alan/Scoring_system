import streamlit as st
import pandas as pd
import os
from alecapp import extract_lines_from_txt, evaluate_scores

# Streamlit app
st.title("Model Evaluation with Pre-Uploaded Files")

# Define paths
results_folder = "results/wmt22/cs-uk"
data_folder = "data/wmt22/cs-uk"

# Sort and find models
models = [folder for folder in os.listdir(results_folder) if os.path.isdir(os.path.join(results_folder, folder))]
models_sorted = sorted(models, key=lambda x: int(x.split('_')[-1]))  # Extract number and sort numerically

# Display available scoring methods
#score_method = ['sacreBLUE','BLEURT', 'Comet']
score_method = ['sacreBLUE']
selected_score_method = st.selectbox("Select a score method:", score_method)

# Load result files for each model
result_files_in_model = {}
for model in models_sorted:
    model_path = os.path.join(results_folder, model)
    result_files_in_model[model] = sorted(
        [result_file for result_file in os.listdir(model_path) if "training" not in result_file],  # Only test set files
        key=lambda x: int(x.split('.')[-3]) if x.split('.')[-3].isdigit() else float('inf')
    )[:5]

# Human translations
reference_files = {
    "test.cs": os.path.join(data_folder, "test.cs"),
    "test.uk": os.path.join(data_folder, "test.uk")
}

# Dataframe for storing scores
columns = ["Model"] + [f"n-best {i}" for i in range(1, 6)]
score_df = pd.DataFrame(columns=columns)


# Loop through models and compute scores
for model in models_sorted:
    scores_row = {"Model": model}

    for file_index, file_name in enumerate(result_files_in_model[model], start=1):
        file_path = os.path.join(results_folder, model, file_name)
        
        with open(file_path, "r", encoding="utf-8") as f:
            machine_translation = f.read()
        
        # Split into sentences
        machine_sentences = extract_lines_from_txt(machine_translation)

        # Load the reference translation
        humanTranslation = "test.uk"  
        reference_file_path = reference_files.get(humanTranslation)
        if reference_file_path:
            with open(reference_file_path, "r", encoding="utf-8") as f:
                reference_translation = f.read()
            reference_sentences = extract_lines_from_txt(reference_translation)

        # Load source sentences for COMET
        # source_sentences = None  
        # source_file_path = reference_files.get("test.cs")
        # if source_file_path:
        #     with open(source_file_path, "r", encoding="utf-8") as f:
        #         source_translation = f.read()
        #     source_sentences = split_into_sentences([source_translation])

        # Evaluate scores
        result = evaluate_scores(machine_sentences, reference_sentences)
        
        # if result:
        #     bleurt_scores, scarebleu_scores, comet_scores, avg_bleurt, avg_scarebleu, avg_comet, source_sentences_used = result
        # else:
        #     bleurt_scores, scarebleu_scores, comet_scores = [], [], []
        if result:
            scarebleu_scores,avg_scarebleu= result
        else:
            scarebleu_scores= []
        # Assign scores to the correct method
        if selected_score_method == "BLEURT":
            scores_row[f"n-best {file_index}"] = bleurt_scores[file_index-1] if bleurt_scores else None
        elif selected_score_method == "sacreBLUE":
            scores_row[f"n-best {file_index}"] = scarebleu_scores[file_index-1] if scarebleu_scores else None
        elif selected_score_method == "Comet" and source_sentences:
            scores_row[f"n-best {file_index}"] = comet_scores[file_index-1] if comet_scores else None

    # Append the row to DataFrame
    score_df = pd.concat([score_df, pd.DataFrame([scores_row])], ignore_index=True)

# Display the dataframe with scores
st.dataframe(score_df)
