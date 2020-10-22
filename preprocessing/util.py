import re

import numpy as np
import pandas as pd
import contractions
from string import punctuation
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException


def read_csv(path_dict):
    df_text = pd.read_csv(f"{path_dict.input_root}/dataset/yelp/{path_dict.text_file_name}")
    df_label = pd.read_csv(f"{path_dict.input_root}/dataset/yelp/{path_dict.label_file_name}")
    return df_text, df_label


def save_text(texts, file_key, args):
    save_path = f"{args.save_root_path}/dataset/yelp/{file_key}.txt"
    with open(save_path, "w", encoding="utf-8") as f:
        for elm in texts:
            f.writelines(f"{elm}\n")


def remove_crlf(reviews, is_output, args) -> list:
    reviews = [review.replace("\r", " ").replace("\n", " ") for review in reviews]
    reviews = [re.sub("( ){2,}", " ", review) for review in reviews]
    if is_output:
        save_text(reviews, "10.remove_crlf", args)
    return reviews


def expand_contractions(reviews, is_output, args) -> list:
    reviews = [contractions.fix(review) for review in reviews]
    if is_output:
        save_text(reviews, "20.expand_contractions", args)
    return reviews


def filter_into_lang_reviews(reviews, labels, lang_code, is_output, args):
    lang_codes = get_lang_codes(reviews)
    reviews = [rev for rev, l in zip(reviews, lang_codes) if l == lang_code]
    labels = [lab for lab, l in zip(labels, lang_codes) if l == lang_code]
    if is_output:
        save_text(reviews, "30.filter_by_lang_text", args)
        save_text(labels, "30.filter_by_lang_label", args)

    return reviews, labels


def get_lang_codes(reviews):
    # Convert character's code into latin-1 to enable language detector to deal with reviews
    reviews_latin = []
    for review in reviews:
        reviews_latin.append(review.encode("latin-1").decode('utf-8'))

    # Categorise reviews into a language code
    lang_codes = []
    for i in range(len(reviews_latin)):
        try:
            lg = detect(reviews_latin[i])
            lang_codes.append(lg)
        except LangDetectException as e:
            lang_codes.append("INVALID")

    return lang_codes


def remove_urls(reviews, is_output, args):
    # Remove URL
    ptn = r"(https?)(:\/\/[-_\.!~*\'()a-zA-Z0-9;\/?:\@&=\+\$,%#]+)"
    reviews = [re.sub(ptn, "", review) for review in reviews]
    if is_output:
        save_text(reviews, "40.remove_urls", args)

    return reviews


def remove_symbols(reviews, is_output, path_dict):
    """ It also do lowercase-nise"""
    # Removing URL makes more than one whitespace, so it reduce them into one whitespace
    reviews = [re.sub("( ){2,}", " ", review) for review in reviews]
    reviews = [''.join([char for char in review.lower() if char not in punctuation]) for review in reviews]
    if is_output:
        save_text(reviews, "50.remove_symbols", path_dict)

    return reviews
