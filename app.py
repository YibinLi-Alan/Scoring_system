import streamlit as st
import os
import sys
from docx import Document
from PyPDF2 import PdfReader
import re
import pandas as pd
from sacrebleu import sentence_bleu
from comet import download_model, load_from_checkpoint

import sys
sys.path.append('/Users/bin/Desktop/scoring_system/bleurt/build/lib/bleurt')
from score import BleurtScorer


# Initialize BLEURT scorer
bleurt_checkpoint = "/Users/bin/Desktop/scoring_system/bleurt/BLEURT-20"
bleurt_scorer = BleurtScorer(bleurt_checkpoint)

# Load COMET model
try:
    comet_model_path = download_model("Unbabel/wmt20-comet-da")
    comet_model = load_from_checkpoint(comet_model_path)
    comet_model.eval()  # Put COMET in evaluation mode
except Exception as e:
    st.warning(f"COMET model could not be loaded: {e}")
    comet_model = None

# Helper functions
def extract_paragraphs_from_docx(file):
    doc = Document(file)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return paragraphs

def extract_paragraphs_from_pdf(file):
    reader = PdfReader(file)
    paragraphs = []
    for page in reader.pages:
        paragraphs.extend(page.extract_text().split("\n"))
    return [p.strip() for p in paragraphs if p.strip()]

def split_into_sentences(paragraphs):
    sentences = []
    for paragraph in paragraphs:
        sentences.extend(re.split(r'(?<=[.!?])\s+|(?<=\u3002|\uff01|\uff1f)', paragraph))
    return [sentence.strip() for sentence in sentences if sentence.strip()]

def evaluate_scores(sentences1, sentences2, source_sentences):
    bleurt_scores = []
    scarebleu_scores = []
    comet_scores = []
    source_sentences_used = []

    # Check if all sentence lists are of the same length
    if len(sentences1) != len(sentences2):
        st.error("The number of sentences in the first and second paragraphs does not match.")
        st.stop()

    if source_sentences and len(sentences1) != len(source_sentences):
        st.error("The number of sentences in the first paragraph does not match the source sentences.")
        st.stop()

    if source_sentences and len(sentences2) != len(source_sentences):
        st.error("The number of sentences in the second paragraph does not match the source sentences.")
        st.stop()

    # Iterate through the sentences and calculate scores
    for i in range(len(sentences1)):
        hyp = sentences1[i]
        ref = sentences2[i]
        src = source_sentences[i] if source_sentences else ""

        try:
            # BLEURT score
            bleurt_score = bleurt_scorer.score(references=[ref], candidates=[hyp])
            bleurt_scores.append(bleurt_score[0] if bleurt_score else 0)
        except Exception as e:
            st.warning(f"Error processing BLEURT score for pair ({hyp}, {ref}): {e}")
            bleurt_scores.append(0)

        try:
            # SacréBLEU score
            scarebleu_scores.append(sentence_bleu(hyp, [ref]).score)
        except Exception as e:
            st.warning(f"Error processing SacréBLEU score for pair ({hyp}, {ref}): {e}")
            scarebleu_scores.append(0)

        if comet_model and source_sentences:
            try:
                comet_input = [{"src": src, "mt": hyp, "ref": ref}]
                comet_predictions = comet_model.predict(comet_input)

                # Access system_score or scores
                if hasattr(comet_predictions, "system_score"):
                    comet_scores.append(comet_predictions.system_score)
                elif hasattr(comet_predictions, "scores") and comet_predictions.scores:
                    comet_scores.append(comet_predictions.scores[0])
                else:
                    st.warning(f"Unexpected COMET output structure: {comet_predictions}")
                    comet_scores.append(0)

                source_sentences_used.append(src)
            except Exception as e:
                st.warning(f"Error processing COMET score for pair ({hyp}, {ref}): {e}")
                comet_scores.append(0)
                source_sentences_used.append("")
        else:
            if not comet_model:
                st.warning("COMET model is not loaded; skipping COMET score evaluation.")
            comet_scores.append(0)
            source_sentences_used.append("")

    # Calculate averages
    avg_bleurt = sum(bleurt_scores) / len(bleurt_scores) if bleurt_scores else 0
    avg_scarebleu = sum(scarebleu_scores) / len(scarebleu_scores) if scarebleu_scores else 0
    avg_comet = sum(comet_scores) / len(comet_scores) if comet_scores else 0

    return bleurt_scores, scarebleu_scores, comet_scores, avg_bleurt, avg_scarebleu, avg_comet, source_sentences_used

# Streamlit app
st.title("File Upload and Sentence Matching with Evaluation")

# Upload files
uploaded_file1 = st.file_uploader("Upload the machine-translated file (DOCX or PDF)", type=["docx", "pdf"], key="file1")
uploaded_file2 = st.file_uploader("Upload the human-generated (standard) file (DOCX or PDF)", type=["docx", "pdf"], key="file2")
uploaded_source = st.file_uploader("Upload the source file (DOCX or PDF, optional for COMET)", type=["docx", "pdf"], key="source")

if uploaded_file1 and uploaded_file2:
    st.write("Files successfully uploaded!")
    if uploaded_source:
        st.write("Source file uploaded for COMET evaluation.")

    if st.button("Generate Matching Table and Evaluate Scores"):
        try:
            # Extract paragraphs from uploaded files
            paragraphs1 = extract_paragraphs_from_docx(uploaded_file1) if uploaded_file1.name.endswith("docx") else extract_paragraphs_from_pdf(uploaded_file1)
            paragraphs2 = extract_paragraphs_from_docx(uploaded_file2) if uploaded_file2.name.endswith("docx") else extract_paragraphs_from_pdf(uploaded_file2)
            source_sentences = None

            if uploaded_source:
                source_paragraphs = extract_paragraphs_from_docx(uploaded_source) if uploaded_source.name.endswith("docx") else extract_paragraphs_from_pdf(uploaded_source)
                source_sentences = split_into_sentences(source_paragraphs)

            # Split paragraphs into sentences
            sentences1 = split_into_sentences(paragraphs1)
            sentences2 = split_into_sentences(paragraphs2)

            # Evaluate scores
            result = evaluate_scores(sentences1, sentences2, source_sentences)

            bleurt_scores, scarebleu_scores, comet_scores, avg_bleurt, avg_scarebleu, avg_comet, source_sentences_used = result

            # Create dataframe for display
            if uploaded_source:
                matched_sentences = list(zip(source_sentences_used, sentences1, sentences2))
                df = pd.DataFrame(matched_sentences, columns=["Source Sentences", "Machine-Generated Sentences", "Standard Translated Sentences"])
            else:
                matched_sentences = list(zip(sentences1, sentences2))
                df = pd.DataFrame(matched_sentences, columns=["Machine-Generated Sentences", "Standard Translated Sentences"])

            df["BLEURT Score"] = bleurt_scores
            df["SacréBLEU Score"] = scarebleu_scores

            if uploaded_source:
                df["COMET Score"] = comet_scores

            # Add row for averages
            avg_row = {
                "Machine-Generated Sentences": "AVERAGE",
                "Standard Translated Sentences": "AVERAGE",
                "BLEURT Score": avg_bleurt,
                "SacréBLEU Score": avg_scarebleu,
            }

            if uploaded_source:
                avg_row["Source Sentences"] = "AVERAGE"
                avg_row["COMET Score"] = avg_comet

            df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

            # Display the results
            st.dataframe(df)
        except Exception as e:
            st.error(f"An error occurred: {e}")
