from transformers import RobertaTokenizerFast
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import BertTokenizer

VOCAB_PATH = "./synthBERT/vocab.txt"

# load vocab file to list
vocab_file = open(VOCAB_PATH, "r")
vocab = vocab_file.read().split("\n")

# use our pretrained tokenizer
# tokenizer = RobertaTokenizerFast.from_pretrained("./synthBERT")
#tokenizer = Tokenizer(WordLevel.from_file("./synthBERT/vocab.txt"))
#tokenizer = BertTokenizer(VOCAB_PATH, do_basic_tokenize=True, additional_special_tokens=vocab)
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_basic_tokenize=True, never_split=vocab)
tokenizer.add_tokens(vocab)
#tokenizer.pre_tokenizer = WhitespaceSplit()

new_tokenizer = tokenizer.basic_tokenizer

output = tokenizer.tokenize("Viral_sinusitis_(disorder) History_of_appendectomy")

print(output)