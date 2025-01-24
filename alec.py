import streamlit as st
from app import extract_lines_from_txt, split_into_sentences, evaluate_scores
import pandas as pd

# Streamlit app
st.title("Alec's Streamlit")

# File uploaders
uploaded_file1 = st.file_uploader("Upload the machine-translated file (TXT)", type=["txt"], key="file1")
uploaded_file2 = st.file_uploader("Upload the human-generated (standard) file (TXT)", type=["txt"], key="file2")
uploaded_source = st.file_uploader("Upload the source file (TXT, optional for COMET)", type=["txt"], key="source")

# Process uploaded files
if uploaded_file1 and uploaded_file2:
    st.write("Files successfully uploaded!")
    if uploaded_source:
        st.write("Source file uploaded for COMET evaluation.")

    # Checkboxes for separate displays
    display_sentences = st.checkbox("Display Matched Sentences")
    display_scores = st.checkbox("Display Evaluation Scores")

    if display_sentences or display_scores:
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
                sentence_df = pd.DataFrame(matched_sentences, columns=["Source Sentences", "Machine-Generated Sentences", "Standard Translated Sentences"])
            else:
                matched_sentences = list(zip(sentences1, sentences2))
                sentence_df = pd.DataFrame(matched_sentences, columns=["Machine-Generated Sentences", "Standard Translated Sentences"])

            # Add scores if the user wants to see them
            if display_scores:
                sentence_df["BLEURT Score"] = bleurt_scores
                sentence_df["SacréBLEU Score"] = scarebleu_scores
                if uploaded_source:
                    sentence_df["COMET Score"] = comet_scores

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

                sentence_df = pd.concat([sentence_df, pd.DataFrame([avg_row])], ignore_index=True)

            # Display sections based on checkboxes
            #.iloc used instead of just writing each index seperately incase more are added later
            if display_sentences:
                st.subheader("Matched Sentences")
                st.dataframe(sentence_df.iloc[:, :3] if uploaded_source else sentence_df.iloc[:, :2])

            if display_scores:
                st.subheader("Evaluation Scores")
                score_columns = ["BLEURT Score", "SacréBLEU Score"]
                if uploaded_source:
                    score_columns.append("COMET Score")
                st.dataframe(sentence_df[score_columns])
                
        except Exception as e:
            st.error(f"An error occurred: {e}")
