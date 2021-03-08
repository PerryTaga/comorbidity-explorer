from tokenizers import Tokenizer
from tokenizers import pre_tokenizers
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

class WordLevelBertTokenizer(BaseTokenizer):
    """ WordLevelBertTokenizer
    Represents a simple word level tokenization for BERT.
    """

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        unk_token: Union[str, AddedToken] = "<unk>",
        sep_token: Union[str, AddedToken] = "<sep>",
        cls_token: Union[str, AddedToken] = "<cls>",
        pad_token: Union[str, AddedToken] = "<pad>",
        mask_token: Union[str, AddedToken] = "<mask>",
        lowercase: bool = False,
        unicode_normalizer: Optional[str] = None,
    ):
        if vocab_file is not None:
            tokenizer = Tokenizer(WordLevel(vocab_file))
        else:
            tokenizer = Tokenizer(WordLevel())

        # Let the tokenizer know about special tokens if they are part of the vocab
        if tokenizer.token_to_id(str(unk_token)) is not None:
            tokenizer.add_special_tokens([str(unk_token)])
        if tokenizer.token_to_id(str(sep_token)) is not None:
            tokenizer.add_special_tokens([str(sep_token)])
        if tokenizer.token_to_id(str(cls_token)) is not None:
            tokenizer.add_special_tokens([str(cls_token)])
        if tokenizer.token_to_id(str(pad_token)) is not None:
            tokenizer.add_special_tokens([str(pad_token)])
        if tokenizer.token_to_id(str(mask_token)) is not None:
            tokenizer.add_special_tokens([str(mask_token)])

        # Check for Unicode normalization first (before everything else)
        normalizers = []

        if unicode_normalizer:
            normalizers += [unicode_normalizer_from_str(unicode_normalizer)]

        if lowercase:
            normalizers += [Lowercase()]

        # Create the normalizer structure
        if len(normalizers) > 0:
            if len(normalizers) > 1:
                tokenizer.normalizer = Sequence(normalizers)
            else:
                tokenizer.normalizer = normalizers[0]

        tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

        if vocab_file is not None:
            sep_token_id = tokenizer.token_to_id(str(sep_token))
            if sep_token_id is None:
                raise TypeError("sep_token not found in the vocabulary")
            cls_token_id = tokenizer.token_to_id(str(cls_token))
            if cls_token_id is None:
                raise TypeError("cls_token not found in the vocabulary")

            tokenizer.post_processor = processors.BertProcessing(
                (str(sep_token), sep_token_id), (str(cls_token), cls_token_id)
            )

        parameters = {
            "model": "WordLevel",
            "unk_token": unk_token,
            "sep_token": sep_token,
            "cls_token": cls_token,
            "pad_token": pad_token,
            "mask_token": mask_token,
            "lowercase": lowercase,
            "unicode_normalizer": unicode_normalizer,
        }

        super().__init__(tokenizer, parameters)