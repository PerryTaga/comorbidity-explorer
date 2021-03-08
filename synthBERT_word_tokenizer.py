# utility to tokenize novel medical terminology from Synthea

VOCAB_PATH = "./synthBERT/vocab.txt"
CORPUS_PATH = "./corpus/corpus-small.txt"
OUTPUT_PATH = "./synthBERT"

"""
available corpora:
corpus-small.txt
corpus-large.txt
"""
corpus_paths = ["./corpus/corpus-small.txt"]

# convert corpus to vocab of unique words
vocab = []
with open(CORPUS_PATH, "r") as f:
	vocab = f.read().split()
vocab = list(dict.fromkeys(vocab))

# add special tokens to vocab
special_tokens = ["[UNK]", "[SEP]","[PAD]","[CLS]","[MASK]"]
special_tokens.extend(vocab)
vocab = special_tokens

# save the vocab
with open(VOCAB_PATH, 'w') as fp:
	for word in vocab[:-1]:
		fp.write(word+"\n")
	# leave off final newline
	fp.write(vocab[-1])

print("Saved to: " + OUTPUT_PATH)