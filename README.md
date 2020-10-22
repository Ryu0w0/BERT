# BERT
A PyTorch implementation of BERT

## How to run
### Pre-processing
1. Get data from [Yelp Dataset in kaggle](https://www.kaggle.com/yelp-dataset/yelp-dataset?select=yelp_academic_dataset_review.json)
2. Put the obtained file `yelp_academic_dataset_review.json` at ./files/input/dataset/yelp
3. Modify the argument `review_num` in `preprocessing_main.py` to the number of reviews that you want to use
4. Run `preprocessing_main.py` and then `yelp_train.tsv` is produced at ./files/load/dataset\yelp\yelp_train.tsv

### Binary sentiment classification
1. Get pre-trained weights from [git repository of BERT](https://github.com/google-research/bert). Please use "4/512 (BERT-Small)".
2. After installing transformer, convert pre-trained weights of TensorFlow into PyTorch by following [Converting Tensorflow Checkpoints](https://huggingface.co/transformers/converting_tensorflow_models.html#bert) and get the converted file of `pytorch_model.bin`
3. `Locate pytorch_model.bin` into ./files/load/model
4. Locate `vocab.txt` in the folder obtained in "1." at "./files/load/vocab"
5. Run `train_bert_main.py`