# BISON : BM25-weighted Self-Attention Network for Multi-Fields Document Search
This is the impletement of paper BISON : BM25-weighted Self-Attention Network for Multi-Fields Document Search https://arxiv.org/abs/2007.05186. Taking MS Marco Document Ranking task as an example.

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

### Data Preprocess Example command
```
mpirun -np 8 python preprocessor_multipreprocess_msmarco_doc.py \
--task_name=BISON \
--data_dir=your_data_folder_path \
--data_file=your_data_file_name \
--data_line_count=your_data_line_count \
--output_dir=your_output_folder_path \
--vocab_file=your_vocab_file \
--full_word_idf_file=your_idf_file \
--default_idf=your_default_idf \
--do_lower_case=True \
--max_seq_length_query=20 --max_seq_length_url=30 --max_seq_length_title=30 \
--label_col=0 --src_col=2 --url_col=4 --title_col=5 \
--BM25_K1=0.25 --BM25_B=0.4 --BM25_AVGDL_Q=6 --BM25_AVGDL_D=25
```

### Training Example command
```
TF_XLA_FLAGS=--tf_xla_auto_jit=2 mpirun --verbose --allow-run-as-root -np 8 --display-map -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x HOROVOD_GPU_ALLREDUCE=NCCL -x PATH -mca pml ob1 -mca btl self,tcp,openib -mca btl_tcp_if_exclude lo,docker0 \
python train_msmarco_doc.py \
--task_name=BISON \
--do_train=True \
--preprocess_train_dir=your_preprocessed_training_data_folder_path \
--train_line_count=your_data_line_count \
--train_partition_count=your_data_partition_count \
--preprocess_train_file_name=your_preprocess_train_file_name \
--preprocess_eval_dir=your_preprocessed_eval_data_folder_path \
--output_dir=your_output_folder_path \
--query_bert_config_file=your_query_config \
--meta_bert_config_file=your_meta_config \
--max_seq_length_query=20 --max_seq_length_url=30 --max_seq_length_title=30 \
--train_batch_size=512 --learning_rate=8e-5 --num_train_epochs=5 --save_checkpoints_steps=800 --eval_batch_size=500 \
--nce_temperature=1000 --nce_weight=0.5 \
--use_fp16=True --use_xla=True --horovod=True  --use_one_hot_embeddings=False --verbose_logging=True
```
