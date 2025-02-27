import streamlit as st
import pandas as pd
import os
import torch
from alecapp import extract_lines_from_txt, evaluate_scores
from comet import download_model, load_from_checkpoint
from bleurt import score

# Streamlit app
st.title("Model Evaluation with Pre-Uploaded Files")

# Define paths
results_folder = "results/wmt22/cs-uk"
data_folder = "data/wmt22/cs-uk"

# Sort and find models
models = sorted(
    [folder for folder in os.listdir(results_folder) if os.path.isdir(os.path.join(results_folder, folder))],
    key=lambda x: int(x.split('_')[-1])  # Extract number and sort numerically
)

# Display available scoring methods
# score_method = ['sacreBLUE', 'BLEURT', 'Comet']
score_method = ['sacreBLUE', 'Comet']
selected_score_method = st.selectbox("Select a score method:", score_method)

# Human translations
reference_file_path = os.path.join(data_folder, "test.uk")
with open(reference_file_path, "r", encoding="utf-8") as f:
    reference_sentences = extract_lines_from_txt(f.read())

# Load source sentences for COMET
source_file_path = os.path.join(data_folder, "test.cs")
source_sentences = None
if source_file_path:
    with open(source_file_path, "r", encoding="utf-8") as f:
        source_sentences = extract_lines_from_txt(f.read())

# Load COMET model
try:
    model_path = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(model_path)
    comet_model.eval()
except Exception as e:
    st.warning(f"COMET model could not be loaded: {e}")
    comet_model = None

# Load BLEURT model
# try:
#     bleurt_checkpoint = "bleurt/BLEURT-20"
#     bleurt_scorer = score.BleurtScorer(bleurt_checkpoint)
# except Exception as e:
#     st.warning(f"BLEURT model could not be loaded: {e}")
#     bleurt_scorer = None

# Get the max number of n-best files (assuming up to 5)
n_best_range = range(1, 6)
columns = ["Model"] + [f"n-best {i}" for i in n_best_range]

# Initialize empty DataFrame
score_df = pd.DataFrame(index=models, columns=columns)
score_df["Model"] = models  # Assign model names

# Loop through models and compute scores
for model in models:
    model_path = os.path.join(results_folder, model)
    result_files = sorted(
        [file for file in os.listdir(model_path) if "training" not in file],  # Only test set files
        key=lambda x: int(x.split('.')[-3]) if x.split('.')[-3].isdigit() else float('inf')
    )[:5]  # Keep up to 5 files

    for file_index, file_name in enumerate(result_files, start=1):
        file_path = os.path.join(model_path, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            machine_sentences = extract_lines_from_txt(f.read())

        # Evaluate score
        scarebleu_scores, avg_scarebleu, comet_scores, avg_comet = evaluate_scores(machine_sentences, reference_sentences, source_sentences)

        # Evaluate BLEURT score
        # if bleurt_scorer:
        #     try:
        #         bleurt_scores = bleurt_scorer.score(references=reference_sentences, candidates=machine_sentences)
        #         avg_bleurt = sum(bleurt_scores) / len(bleurt_scores)
        #     except Exception as e:
        #         st.warning(f"Error processing BLEURT score for {file_name}: {e}")
        #         avg_bleurt = None
        # else:
        #     avg_bleurt = None

        # # Evaluate COMET score
        # if comet_model and source_sentences:
        #     try:
        #         comet_input = [{"src": src, "mt": mt, "ref": ref} for src, mt, ref in zip(source_sentences, machine_sentences, reference_sentences)]
        #         if torch.cuda.is_available():
        #             comet_predictions = comet_model.predict(comet_input, batch_size=8, gpus=1)
        #         else:
        #             comet_model.device = torch.device("cpu")
        #             comet_predictions = comet_model.predict(comet_input, batch_size=1)

        #         # Extract scores
        #         if hasattr(comet_predictions, "system_score"):
        #             avg_comet = comet_predictions.system_score
        #         elif hasattr(comet_predictions, "scores") and comet_predictions.scores:
        #             avg_comet = sum(comet_predictions.scores) / len(comet_predictions.scores)
        #         else:
        #             st.warning(f"Unexpected COMET output structure for {file_name}: {comet_predictions}")
        #             avg_comet = None
        #     except Exception as e:
        #         st.warning(f"Error processing COMET score for {file_name}: {e}")
        #         avg_comet = None
        # else:
        #     avg_comet = None

        # Assign scores to DataFrame based on selection
        # if selected_score_method == "BLEURT":
        #     score_df.at[model, f"n-best {file_index}"] = avg_bleurt
        if selected_score_method == "sacreBLUE":
            score_df.at[model, f"n-best {file_index}"] = avg_scarebleu
        elif selected_score_method == "Comet":
            score_df.at[model, f"n-best {file_index}"] = avg_comet

# Display the dataframe with scores
st.dataframe(score_df)
