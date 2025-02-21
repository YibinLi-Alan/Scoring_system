import streamlit as st
from sacrebleu.metrics import BLEU
from comet import download_model, load_from_checkpoint
from bleurt import score


bleu = BLEU()
bleurt_checkpoint = "bleurt/BLEURT-20"
try:
    scorer = score.BleurtScorer(bleurt_checkpoint)
except Exception as e:
    st.error(f"Failed to load BLEURT model: {e}")
    scorer = None 


def evaluete_scores_scarebleu(sentences1, sentences2):
    scarebleu_scores = []
    for i in range(len(sentences1)):
        hyp = sentences1[i]
        ref = sentences2[i]

        try:
            # SacréBLEU score
            scarebleu_scores.append(bleu.corpus_score([hyp], [[ref]]).score)
        except Exception as e:
            st.warning(f"Error processing SacréBLEU score for pair ({hyp}, {ref}): {e}")
            scarebleu_scores.append(0)
    avg_scarebleu = sum(scarebleu_scores) / len(scarebleu_scores) if scarebleu_scores else 0
    return scarebleu_scores, avg_scarebleu


def evaluete_scores_bluert(sentences1, sentences2):
    hyp = sentences1
    ref = sentences2
    try: 
        score_value = scorer.score(references=ref, candidates=hyp)
        avg_bleurt = sum(score_value) / len(score_value) if score_value else 0
        score_value = [str(s) for s in score_value]  # Convert to strings
        print(score_value)
    except Exception as e:
        print(f"Error processing BLEURT score: {e}")
        return 0, ["NA"]
    return avg_bleurt, score_value



model_path = download_model("Unbabel/wmt22-comet-da")
model = load_from_checkpoint(model_path)

def evaluete_scores_comet(sentences1, sentences2, source_sentences):
    comet_scores = []
    hyp = sentences1
    ref = sentences2
    src = source_sentences
    try:
        comet_input = [{"src": src, "mt": hyp, "ref": ref}]
        comet_scores = model.predict(comet_input, batch_size=8, gpus=1)
    except Exception as e:
        st.warning(f"Error processing COMET score for pair ({hyp}, {ref}): {e}")
        comet_scores.append(0)
    avg_comet = sum(comet_scores) / len(comet_scores) if comet_scores else 0
    return comet_scores, avg_comet
    



    

# def evaluate_scores(sentences1, sentences2, source_sentences):
#     bleurt_scores = []
#     scarebleu_scores = []
#     comet_scores = []
#     source_sentences_used = []

#     checkpoint = "bleurt/BLEURT-20"
#     scorer = score.BleurtScorer(checkpoint)
#     # Iterate through the sentences and calculate scores
#     for i in range(len(sentences1)):
#         hyp = sentences1[i]
#         ref = sentences2[i]
#         src = source_sentences[i] if source_sentences else ""

#         try:
#             # BLEURT score
#             checkpoint = "bleurt/BLEURT-20"
#             scorer = score.BleurtScorer(checkpoint)
#             bleurt_scores= scorer.score(references=[ref], candidates=[hyp])
#         except Exception as e:
#             st.warning(f"Error processing BLEURT score for pair ({hyp}, {ref}): {e}")
#             bleurt_scores.append(0)

#         try:
#             # SacréBLEU score
#             scarebleu_scores.append(bleu.corpus_score([hyp], [[ref]]).score)
#         except Exception as e:
#             st.warning(f"Error processing SacréBLEU score for pair ({hyp}, {ref}): {e}")
#             scarebleu_scores.append(0)

#         if comet_model and source_sentences:
#             try:
#                 comet_input = [{"src": src, "mt": hyp, "ref": ref}]
#                 if torch.cuda.is_available():
#                 # Use GPU for prediction
#                     comet_predictions = comet_model.predict(comet_input, batch_size=8, gpus=1)
#                 else:
#                 # Use CPU for prediction
#                     comet_model.device = torch.device("cpu")
#                     comet_predictions = comet_model.predict(comet_input, batch_size=1)

#                 # Access system_score or scores
#                 if hasattr(comet_predictions, "system_score"):
#                     comet_scores.append(comet_predictions.system_score)
#                 elif hasattr(comet_predictions, "scores") and comet_predictions.scores:
#                     comet_scores.append(comet_predictions.scores[0])
#                 else:
#                     st.warning(f"Unexpected COMET output structure: {comet_predictions}")
#                     comet_scores.append(0)

#                 source_sentences_used.append(src)
#             except Exception as e:
#                 st.warning(f"Error processing COMET score for pair ({hyp}, {ref}): {e}")
#                 comet_scores.append(0)
#                 source_sentences_used.append("")
#         else:
#             if not comet_model:
#                 st.warning("COMET model is not loaded; skipping COMET score evaluation.")
#             comet_scores.append(0)
#             source_sentences_used.append("")

#     # Calculate averages
#     avg_bleurt = sum(bleurt_scores) / len(bleurt_scores) if bleurt_scores else 0
#     avg_scarebleu = sum(scarebleu_scores) / len(scarebleu_scores) if scarebleu_scores else 0
#     avg_comet = sum(comet_scores) / len(comet_scores) if comet_scores else 0

#     return bleurt_scores, scarebleu_scores, comet_scores, avg_bleurt, avg_scarebleu, avg_comet, source_sentences_used

