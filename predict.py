from transformers import pipeline
from transformers import BertTokenizer
from transformers import BertForMaskedLM


def predict(masked_phrase):

	MODEL_FOLDER = "./synthBERT"
	VOCAB_PATH = "./synthBERT/vocab.txt"

	# load vocab file to list
	vocab_file = open(VOCAB_PATH, "r")
	vocab = vocab_file.read().split("\n")

	# use our pretrained tokenizer and model
	tokenizer = BertTokenizer(VOCAB_PATH, do_basic_tokenize=True, additional_special_tokens=vocab)
	#tokenizer.add_tokens(vocab)
	model = BertForMaskedLM.from_pretrained(MODEL_FOLDER)

	fill_mask = pipeline(
	    "fill-mask",
	    model= model,
	    tokenizer=tokenizer,
	    topk = 10
	)

	#predictions = fill_mask("Viral_sinusitis_(disorder) [MASK]")
	predictions = fill_mask(masked_phrase)

	split_predictions = []
	for prediction in predictions:
		split_predictions.append([prediction['token_str'],	str(prediction['score']*100)])

	# returns ordered list of [predicted_condition, confidence_precent]
	return split_predictions