# tokenizer imports
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.normalizers import NFD
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.implementations import BaseTokenizer

# model imports
from transformers import BertConfig
from transformers import BertTokenizer
from transformers import BertForMaskedLM
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer
from transformers import TrainingArguments
from transformers import logging

CORPUS_PATH = "./corpus/corpus-small.txt"
VOCAB_PATH = "./synthBERT/vocab.txt"
OUTPUT_DIRECTORY = "./synthBERT"

# load vocab file to list
vocab_file = open(VOCAB_PATH, "r")
vocab = vocab_file.read().split("\n")

# use our pretrained tokenizer and model
tokenizer = BertTokenizer(VOCAB_PATH, do_basic_tokenize=True, additional_special_tokens=vocab)
#tokenizer.add_tokens(vocab)

# set BERT model parameters
config = BertConfig(
	vocab_size = 141,
	max_position_embeddings = 50,
	num_addention_heads = 12,
	num_hidden_layers = 6,
)

# instantiate the model
# transformers has some built specifically for masked language modeling
model = BertForMaskedLM(config=config)

# resize the model embedding to fit our own vocab
model.resize_token_embeddings(len(tokenizer))

# put corpus into a dataset helper
dataset = LineByLineTextDataset(
	tokenizer = tokenizer,
	file_path = CORPUS_PATH,
	block_size = 128,
)

# instantiate a helper to split the dataset when needed in the model
data_collator = DataCollatorForLanguageModeling(
	tokenizer = tokenizer,
	mlm = True,
	mlm_probability = 0.15,
)

# setup training log settings
logging.set_verbosity_info()

# set trainer parameters
training_args = TrainingArguments(
	output_dir = OUTPUT_DIRECTORY,
	overwrite_output_dir = True,
	num_train_epochs = 10,
	per_device_train_batch_size = 129,
	save_steps = 100,
	save_total_limit = 2,
	prediction_loss_only = True,
	logging_steps = 1,
	do_train = True,
	learning_rate = 10e-5
)

# create a trainer object to do the training
trainer = Trainer(
	model = model,
	args = training_args,
	data_collator = data_collator,
	train_dataset = dataset,
)

# train and save the model
trainer.train()
trainer.save_model(OUTPUT_DIRECTORY)

print("Success!")
print("Saved model to: " + OUTPUT_DIRECTORY)