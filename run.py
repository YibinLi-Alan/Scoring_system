import streamlit as st
from app import extract_lines_from_txt, split_into_sentences,evaluate_scores
import pandas as pd
# Streamlit app
st.title("TXT File Upload and Sentence Matching with Evaluation")

# Upload files
uploaded_file1 = st.file_uploader("Upload the machine-translated file (TXT)", type=["txt"], key="file1")
uploaded_file2 = st.file_uploader("Upload the human-generated (standard) file (TXT)", type=["txt"], key="file2")
uploaded_source = st.file_uploader("Upload the source file (TXT, optional for COMET)", type=["txt"], key="source")

if uploaded_file1 and uploaded_file2:
    st.write("Files successfully uploaded!")
    if uploaded_source:
        st.write("Source file uploaded for COMET evaluation.")

    if st.button("Generate Matching Table and Evaluate Scores"):
        try:
            # Extract lines from uploaded files
            lines1 = extract_lines_from_txt(uploaded_file1)
            lines2 = extract_lines_from_txt(uploaded_file2)
            source_lines = extract_lines_from_txt(uploaded_source) if uploaded_source else None

            # Split lines into sentences
            sentences1 = split_into_sentences(lines1)
            sentences2 = split_into_sentences(lines2)
            source_sentences = split_into_sentences(source_lines) if source_lines else None

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
