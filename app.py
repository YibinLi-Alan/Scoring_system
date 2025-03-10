import streamlit as st
from sacrebleu.metrics import BLEU
from comet import download_model, load_from_checkpoint
from bleurt import score
import json
import os



bleu = BLEU()
bleurt_checkpoint = "bleurt/BLEURT-20"
try:
    scorer = score.BleurtScorer(bleurt_checkpoint)
except Exception as e:
    st.error(f"Failed to load BLEURT model: {e}")
    scorer = None 

CACHE_FILE = "score_cache.json"

if not os.path.exists(CACHE_FILE):
  with open(CACHE_FILE,"w") as f:
      json.dump({},f)

with open(CACHE_FILE,"r") as f:
    try:
        score_cache=json.load(f)
    except json.JSONDecodeError:
        print("error reading cache file")
        score_cache={}


def save_cache():
    """Load existing cache, update it with new scores, and save it back to the file."""
    global score_cache

    # Load existing cache first
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                existing_cache = json.load(f)
        except json.JSONDecodeError:
            existing_cache = {}
    else:
        existing_cache = {}

    # Merge new scores with existing ones
    existing_cache.update(score_cache)

    # Save updated cache
    with open(CACHE_FILE, "w") as f:
        json.dump(existing_cache, f, indent=4)

    # Update in-memory cache
    score_cache = existing_cache




def evaluete_scores_scarebleu(sentences1, sentences2,filename):
    cache_key = f"{filename}_sacreBlue"
    if cache_key in score_cache:
        print(f"Loading cached Sacreblue scores for {filename}")
        return score_cache[cache_key]["scores"],score_cache[cache_key]["average"]
    
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
    score_cache[cache_key]={"scores":scarebleu_scores,"average":avg_scarebleu}
    save_cache()
    return scarebleu_scores, avg_scarebleu


def evaluete_scores_bluert(sentences1, sentences2,filename):
    cache_key = f"{filename}_bluert"
    if cache_key in score_cache:
        print(f"Loading cached Sacreblue scores for {filename}")
        cached_scores = score_cache[cache_key]["scores"]
        cached_avg = score_cache[cache_key]["average"]
        if isinstance(cached_scores, float):
            cached_scores = [cached_scores]
        return cached_avg, cached_scores
    
    hyp = sentences1
    ref = sentences2
    
    try: 
        score_value = scorer.score(references=ref, candidates=hyp)
        if isinstance(score_value, float):
            score_value = [score_value]
        avg_bleurt = sum(score_value) / len(score_value) if score_value else 0
        score_value_str = [str(s) for s in score_value]  # Convert to strings
        print(score_value_str)
    except Exception as e:
        print(f"Error processing BLEURT score: {e}")
        return 0, ["NA"]
    score_cache[cache_key]={"scores":score_value_str,"average":avg_bleurt}
    save_cache()
    return avg_bleurt,score_value_str



model_path = download_model("Unbabel/wmt22-comet-da")
model = load_from_checkpoint(model_path)

def evaluete_scores_comet(sentences1, sentences2, source_sentences,filename):
    cache_key = f"{filename}_comet"
    if cache_key in score_cache:
        print(f"Loading cached Sacreblue scores for {filename}")
        return score_cache[cache_key]["scores"],score_cache[cache_key]["average"]
    
    try:
        comet_input = [{"src": src, "mt": hyp, "ref": ref} for src, hyp, ref in zip(source_sentences, sentences1, sentences2)]
        comet_scores = model.predict(comet_input, batch_size=8, gpus=1)
        comet_scores = comet_scores["scores"]
    except Exception as e:
        st.warning(f"Error processing COMET score: {e}")
        comet_scores = [0] * len(sentences1)
    avg_comet = sum(comet_scores) / len(comet_scores) if comet_scores else 0
    score_cache[cache_key]={"scores":comet_scores,"average":avg_comet}
    save_cache()

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

