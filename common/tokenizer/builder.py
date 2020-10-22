from common.util import text_processing
from common.util.text_processing import BertTokenizer


def build_tokenizer(vocab_file):
    """
    It returns a tokenizer function for torchtext.data.Field
    """
    tokenizer_bert = BertTokenizer(vocab_file=vocab_file, do_lower_case=True)

    def preprocessing_and_tokenizating(text, tokenizer=tokenizer_bert.tokenize):
        text = text_processing.preprocessing_text(text)
        text = tokenizer(text)  # tokenizer_bert
        return text

    return preprocessing_and_tokenizating
