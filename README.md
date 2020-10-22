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

## Network structure

The figure below shows the structure of BERT constructed in this repository for the binary sentiment analysis on Yelp dataset. BERT consists of 3 modules which are BERT Embeddings, BERT encoder and BERT Pooler. A classifier followed these structures. 

<img src="https://github.com/Ryu0w0/meta_repository/blob/master/BERT/images/structure.PNG" width=60%>

In a classifier module, an additional fully-connected layer is added (Figure 3). This approach was led by the following data exploration. Firstly, the number of tokens per review was calculated and plotted as histograms shown in Figure 2. The frequency is approximately 24000 around 50 tokens of reviews in the histogram over positive reviews in (a), whereas it is around 16000 for negative reviews in (b). It indicates that shorter reviews are more likely to be positive reviews and vice versa. Although BERT considers the meaning of words and position of words, it does not include the information of the length of sequences. This approach slightly improved the validation accuracy from 0.9491 to 0.9509.

<img src="https://github.com/Ryu0w0/meta_repository/blob/master/BERT/images/histogram.PNG" width=60%>

