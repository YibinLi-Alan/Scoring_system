import sacrebleu

references = ["This is a test."]
candidates = ["This is the test."]

# Initialize the BLEU scorer (you can specify different BLEU variations)
bleu = sacrebleu.metrics.BLEU()  # Default BLEU

# Calculate BLEU score
results = bleu.corpus_score(candidates, references)

# Access the BLEU score (it's a named tuple)
bleu_score = results.score

print(bleu_score)  # Print the BLEU score

# If you want the individual sentence scores (if you have multiple sentences):
bleu_sentence_scores = bleu.sentence_score(candidates, references)
print(bleu_sentence_scores)

# Or if you want to access the details:
print(results)
print(results.sys_len)
print(results.ref_len)
