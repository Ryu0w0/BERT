import collections
import re
import string

from external.utils.tokenizer import BasicTokenizer, WordpieceTokenizer


def load_vocab(vocab_file):
    """Read vocab files into dict"""
    vocab = collections.OrderedDict()
    ids_to_tokens = collections.OrderedDict()
    index = 0

    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()

            vocab[token] = index
            ids_to_tokens[index] = token
            index += 1

    return vocab, ids_to_tokens


def preprocessing_text(text):
    """Pre-processing to each token"""
    # Replace symbols with white space except period and comma
    for p in string.punctuation:
        if (p == ".") or (p == ","):
            continue
        else:
            text = text.replace(p, " ")

    text = text.replace(".", " . ")
    text = text.replace(",", " , ")
    return text


class BertTokenizer(object):
    """  Tokenizer for Bert"""

    def __init__(self, vocab_file, do_lower_case=True):
        """
        vocab_file: str
            file path to vocabulary file
        do_lower_case: boolean
            whether lower case when tokenization
        """
        # load vocabs
        self.vocab, self.ids_to_tokens = load_vocab(vocab_file)

        # define special token symbols which are not tokenized
        never_split = ("[UNK]", "[unk]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")

        # members
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case,
                                              never_split=never_split)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

    def tokenize(self, text):
        split_tokens = []
        # tokenize with whitespace
        for token in self.basic_tokenizer.tokenize(text):
            # tokenize into sub-tokens
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        """ from word(token) to vocab id """
        ids = []
        for token in tokens:
            ids.append(self.vocab[token])

        return ids

    def convert_ids_to_tokens(self, ids):
        """ from vocab id to word(token) """
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens


