import streamlit as st
from sacrebleu.metrics import BLEU
from comet import download_model, load_from_checkpoint
import torch
from bleurt import score

bleu = BLEU(effective_order=True)
# Load COMET model
try:
    model_path = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(model_path)
    comet_model.eval()  # Put COMET in evaluation mode
except Exception as e:
    st.warning(f"COMET model could not be loaded: {e}")
    comet_model = None

# Helper functions
def extract_lines_from_txt(text):
    lines = text.splitlines()
    return [line.strip() for line in lines if line.strip()]

# checkpoint = "../BLEURT-20"
# bleurt_scorer = score.BleurtScorer(checkpoint)
def evaluate_scores(sentences1, sentences2,source_sentences):
    # bleurt_scores = []
    scarebleu_scores = []
    comet_scores = []
    
    scarebleu_scores = [bleu.sentence_score(hyp, [ref]).score for hyp, ref in zip(sentences1, sentences2)]
    avg_scarebleu = bleu.corpus_score(sentences1, [sentences2]).score
    
    # try:
    #     bleurt_scores = bleurt_scorer.score(references=sentences2, candidates=sentences1)
    #     avg_bleurt = sum(bleurt_scores) / len(bleurt_scores) if bleurt_scores else 0
    # except Exception as e:
    #     print(f"Error computing BLEURT scores: {e}")
    #     bleurt_scores = [0] * len(sentences1)  # Default to zero if error
    #     avg_bleurt = 0

    if comet_model and source_sentences:
        try:
            comet_input = [{"src": src, "mt": hyp, "ref": ref} for src, hyp, ref in zip(source_sentences, sentences1, sentences2)]
            if torch.cuda.is_available():
            # Use GPU for prediction
                comet_predictions = comet_model.predict(comet_input, batch_size=8, gpus=1)
            else:
                # Use CPU for prediction
                comet_model.device = torch.device("cpu")
                comet_predictions = comet_model.predict(comet_input, batch_size=1)

                # Access system_score or scores
            comet_scores = comet_predictions.scores if hasattr(comet_predictions, "scores") else [0] * len(sentences1)

        except Exception as e:
            st.warning(f"Error processing COMET score for pair ({hyp}, {ref}): {e}")
            comet_scores.append(0)
    else:
        if not comet_model:
            st.warning("COMET model is not loaded; skipping COMET score evaluation.")
            comet_scores.append(0)

    # Calculate averages
    # avg_bleurt = sum(bleurt_scores) / len(bleurt_scores) if bleurt_scores else 0
    avg_scarebleu = sum(scarebleu_scores) / len(scarebleu_scores) if scarebleu_scores else 0
    avg_comet = sum(comet_scores) / len(comet_scores) if comet_scores else 0

    # return bleurt_scores, scarebleu_scores, comet_scores, avg_bleurt, avg_scarebleu, avg_comet, source_sentences_used
    return scarebleu_scores, avg_scarebleu, comet_scores, avg_comet


