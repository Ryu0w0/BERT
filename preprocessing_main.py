"""
This program is for pre-processing data.
It performs removing change line symbols and URLs, expanding contractions,
 filtering into English reviews, and shifting label values from 1-5 to 0 and 1
"""


def initialization():
    import argparse
    from common.logger import util as log_util

    # parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("-save_key", type=str, default="prep_bert", help="Name of log file.")
    parser.add_argument("-save_root_path", type=str, default="./files/output", help="Location of log file.")
    parser.add_argument("-review_num", type=int, default=500, help="Number of reviews retrieved from original data")
    parser.add_argument("-log_level", type=str, default="INFO")
    parser.add_argument("-is_output", type=int, default=0,
                        help="Flag whether producing temporary files of processed data.")
    args = parser.parse_args()

    # create logger
    args.log_level = log_util.get_log_level_from_name(args.log_level)
    logger_ = log_util.create_logger("main", args.save_root_path, args.save_key, args.log_level)
    log_util.logger_ = logger_

    # logging
    logger_.info("*** ARGUMENTS ***")
    logger_.info(args)

    return args


def main():
    args = initialization()

    import pandas as pd

    from preprocessing import util as pp_util
    from common.logger.util import logger_

    logger_.info("*** PRE-PROCESSING ***")
    # preprocessing
    reader = pd.read_json("./files/input/dataset/yelp/yelp_academic_dataset_review.json",
                          chunksize=args.review_num, lines=True)
    for chunk in reader:
        reviews, labels = chunk["text"].values, chunk["stars"].values
        break
    # Initial data
    if args.is_output:
        pp_util.save_text(reviews, "00.initial_text", args)
        pp_util.save_text(labels, "00.initial_label", args)
    logger_.info(f"Initial: review len: {len(reviews)}, label len:{len(labels)}")

    # Remove change line symbols
    reviews = pp_util.remove_crlf(reviews, is_output=args.is_output, args=args)
    logger_.info(f"Aft remove_crlf: review len: {len(reviews)}, label len:{len(labels)}")

    # Expand contractions
    reviews = pp_util.expand_contractions(reviews, is_output=args.is_output, args=args)
    logger_.info(f"Aft expand_contractions: review len: {len(reviews)}, label len:{len(labels)}")

    # Filter into English reviews
    # reviews, labels = pp_util.filter_into_lang_reviews(reviews, labels, "en", is_output=args.is_output, args=args)
    # logger_.info(f"Aft filter_into_lang_reviews: review len: {len(reviews)}, label len:{len(labels)}")

    # Remove URL
    reviews = pp_util.remove_urls(reviews, is_output=args.is_output, args=args)
    logger_.info(f"Aft remove_urls: review len: {len(reviews)}, label len:{len(labels)}")
    # reviews = pp_util.remove_symbols(reviews, is_output=args.is_output, path_dict=path_dict)

    # Convert labels from star rates 1 - 5 into class idx 0 to 4
    op_path = f"./files/load/dataset/yelp/yelp_train.tsv"
    df = pd.DataFrame({"reviews": reviews, "labels": labels})
    # drop rows of neutral sentiment
    df = df[df["labels"].isin([1, 2, 4, 5])]
    # convert sentiments of 1 and 2 into 0 (negative)
    df.loc[df["labels"].isin([1, 2]), "labels"] = 0
    # convert sentiments of 4 and 5 into 1 (positive)
    df.loc[df["labels"].isin([4, 5]), "labels"] = 1
    # Save to load folder
    df.to_csv(op_path, sep='\t', index=False, header=False)
    logger_.info(f"FINISH PRE-PROCESSING")


if __name__ == '__main__':
    main()

