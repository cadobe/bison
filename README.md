# BISON : BM25-weighted Self-Attention Network for Multi-Fields Document Search
This is the impletement of paper BISON : BM25-weighted Self-Attention Network for Multi-Fields Document Search. Take MS Marco Document Ranking task as an example.

## Getting Started
This version is built with a distributed training with horovod approach

## Prerequisites
```
tensorflow>=1.14.0
horovod
```

## Running
The folder "msmarco_doc_preprocess" is responsible for preprocess data of MS Marco.
The folder "msmarco_doc_train" is used to train the data with BISON.
Entrance file is train_msmarco_doc.py
