# coding=utf-8
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import re
import shutil
import math
import collections
import csv
import os
from modeling import modeling_bison
import tensorflow as tf
import horovod.tensorflow as hvd
import time
from utils.utils import LogEvalRunHook, LogTrainRunHook
import numpy as np
from tensorflow.python.client import device_lib
from tensorflow.python.summary.writer import writer_cache
import logging
import sys

flags = tf.flags
FLAGS = flags.FLAGS
tf.logging.set_verbosity(logging.INFO)

# Required parameters
flags.DEFINE_string("task_name", None, "The name of the task to train.")
flags.DEFINE_bool("do_train", False, "Whether to run training.")
flags.DEFINE_string("preprocess_train_dir", None, "The preprocess training data directory.")
flags.DEFINE_integer("train_line_count", None, "Data file line count.")
flags.DEFINE_integer("train_partition_count", None, "Total count of training files.")
flags.DEFINE_string("preprocess_train_file_name", "train.tf_record", "The preprocess training file name.")
flags.DEFINE_string("preprocess_eval_dir", None, "The preprocess eval data directory.")
flags.DEFINE_string("output_dir", None, "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "query_bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "meta_bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

# Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
    "max_seq_length_query", 20,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "max_seq_length_url", 30,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")    

flags.DEFINE_integer(
    "max_seq_length_title", 30,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")  

flags.DEFINE_integer(
    "max_seq_length_body", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.") 

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 32, "Total batch size for eval.")

flags.DEFINE_float("learning_rate", 5e-5,
                   "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")                   

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")
flags.DEFINE_integer("num_accumulation_steps", 1,
                     "Number of accumulation steps before gradient update"
                     "Global batch size = num_accumulation_steps * train_batch_size")
flags.DEFINE_bool("use_fp16", False,
                  "Whether to use fp32 or fp16 arithmetic on GPU.")
flags.DEFINE_bool("use_xla", False, 
                  "Whether to enable XLA JIT compilation.")
flags.DEFINE_bool("horovod", False,
                  "Whether to use Horovod for multi-gpu runs")
flags.DEFINE_bool("use_one_hot_embeddings", False,
                  "Whether to use use_one_hot_embeddings")

flags.DEFINE_float("nce_temperature", 10, "nce_temperature")
flags.DEFINE_float("nce_weight", 0.5, "nce_weight")
flags.DEFINE_string("activation", 'relu', "activation")

flags.DEFINE_bool(
    "verbose_logging", False,
    "If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal SQuAD evaluation.")

flags.DEFINE_bool(
    "Fix_Sim_Weight", False,
    "Whether to fix sim weight and bias.")

flags.DEFINE_integer("sim_weight", None, "sim_weight")

flags.DEFINE_integer("sim_bias", None, "sim_bias")

flags.DEFINE_bool("enable_body", False, "whether to enable body in training.")



def get_func_by_task(task: str):
    """
    Get model builder function, input builder function by task
    """
    if task == "bison":
        from modeling.create_model_bison import model_fn_builder, file_based_input_fn_builder, eval_file_based_input_fn_builder
        return model_fn_builder, file_based_input_fn_builder, eval_file_based_input_fn_builder
    else:
        raise ValueError("Unsupported Task: " + task)


# this function will check how many example in a tfrecord file (used for eval-file)
# file_path must be str or list
def check_line_count_in_tfrecords(file_path):
    line_count = 0

    if isinstance(file_path, str):
        for _ in tf.python_io.tf_record_iterator(file_path):
            line_count += 1
    elif isinstance(file_path, list):
        for f in file_path:
            for _ in tf.python_io.tf_record_iterator(f):
                line_count += 1
    else:
        raise ValueError('file_path must be str or str-list')

    return line_count


# folder: folder path
# files: list
def find_all_file_in_folder(folder, file_paths):
    for file_name in os.listdir(folder):
        path = os.path.join(folder, file_name)

        if os.path.isfile(path):
            file_paths.append(path)
        elif os.path.isdir(path):
            find_all_file_in_folder(path, file_paths)

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    if FLAGS.horovod:
        hvd.init()

    if FLAGS.use_fp16:
        os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE"] = "1"

    model_fn_builder, file_based_input_fn_builder, eval_file_based_input_fn_builder = get_func_by_task(FLAGS.task_name.lower())

    if not FLAGS.do_train:
        raise ValueError("`do_train` must be True.")

    query_bert_config = modeling_bison.BertConfig.from_json_file(
        FLAGS.query_bert_config_file)
    meta_bert_config = modeling_bison.BertConfig.from_json_file(
        FLAGS.meta_bert_config_file)

    # Sequence length check
    if FLAGS.max_seq_length_query > query_bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use query sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length_query, query_bert_config.max_position_embeddings))

    meta_seq_length = FLAGS.max_seq_length_url + FLAGS.max_seq_length_title
    if FLAGS.enable_body:
        meta_seq_length += FLAGS.max_seq_length_body
    if meta_seq_length > meta_bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use meta sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (meta_seq_length, meta_bert_config.max_position_embeddings))
    
    master_process = True
    training_hooks = []
    global_batch_size = FLAGS.train_batch_size * FLAGS.num_accumulation_steps
    hvd_rank = 0
    config = tf.ConfigProto()
    if FLAGS.horovod:
        tf.logging.info("Multi-GPU training with TF Horovod")
        tf.logging.info("hvd.size() = %d hvd.rank() = %d",
                        hvd.size(), hvd.rank())
        global_batch_size = FLAGS.train_batch_size * \
            FLAGS.num_accumulation_steps * hvd.size()
        master_process = (hvd.rank() == 0)
        hvd_rank = hvd.rank()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = str(hvd.local_rank())
        if hvd.size() > 1:
            training_hooks.append(hvd.BroadcastGlobalVariablesHook(0))
    if FLAGS.use_xla:
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    tf.gfile.MakeDirs(FLAGS.output_dir)
    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.output_dir if master_process else None,
        session_config=config,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps if master_process else None,
        keep_checkpoint_max=10)

    if master_process:
        tf.logging.info("***** Configuaration *****")
        for key in FLAGS.__flags.keys():
            tf.logging.info('  {}: {}'.format(key, getattr(FLAGS, key)))
        tf.logging.info("**************************")

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    train_examples_count = FLAGS.train_line_count
    log_train_run_hook = LogTrainRunHook(global_batch_size, hvd_rank)
    training_hooks.append(log_train_run_hook)

    if FLAGS.do_train:
        num_train_steps = int(
            train_examples_count / global_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        query_bert_config=query_bert_config,
        meta_bert_config=meta_bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_one_hot_embeddings=FLAGS.use_one_hot_embeddings,
        nce_temperature=FLAGS.nce_temperature,
        nce_weight=FLAGS.nce_weight,
        hvd=None if not FLAGS.horovod else hvd)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config)

    if FLAGS.do_train:
        start_index = 0
        end_index = FLAGS.train_partition_count

        if FLAGS.horovod:
            tfrecord_per_GPU = int(FLAGS.train_partition_count / hvd.size())
            start_index = hvd.rank() * tfrecord_per_GPU
            end_index = start_index+tfrecord_per_GPU

            if hvd.rank() == hvd.size():
                end_index = FLAGS.train_partition_count

        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", train_examples_count)
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        tf.logging.info("  hvd rank = %d", hvd.rank())
        tf.logging.info("  Num start_index = %d", start_index)
        tf.logging.info("  Num end_index = %d", end_index)

        train_file_list = []
        for i in range(start_index, end_index):
            train_file_list.append(os.path.join(
                FLAGS.preprocess_train_dir, str(i), FLAGS.preprocess_train_file_name))
        tf.logging.info("merge "+str(end_index-start_index) +
                        " preprocessed file from preprocess dir")
        tf.logging.info(train_file_list)

        train_input_fn = file_based_input_fn_builder(
            input_file=train_file_list,
            batch_size=FLAGS.train_batch_size,
            query_seq_length=FLAGS.max_seq_length_query,
            meta_seq_length=meta_seq_length,
            is_training=True,
            drop_remainder=True,
            is_fidelity_eval=False,
            hvd=None if not FLAGS.horovod else hvd)

        # initilize eval file
        # must set preprocess_eval_dir, all file in folder preprocess_eval_dir will thinked as tfrecord file
        if FLAGS.preprocess_eval_dir is None:
            raise ValueError('must set preprocess_eval_dir by hand.')

        all_eval_files = []
        eval_file_list = []
        find_all_file_in_folder(FLAGS.preprocess_eval_dir, all_eval_files)
        for i in range(len(all_eval_files)):
            if hvd.rank() == i % hvd.size():
                eval_file_list.append(all_eval_files[i])

        if 0 == len(eval_file_list):
            raise ValueError('  Rank: %d get eval file empty.' % (hvd.rank()))
        
        tf.logging.info("**********Check how many eval example in current rank*************")
        eval_examples_count = check_line_count_in_tfrecords(eval_file_list)

        eval_steps = int(math.ceil(eval_examples_count / FLAGS.eval_batch_size))

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Rank: %d will eval files:%s" %
                        (hvd.rank(), str(eval_file_list)))
        tf.logging.info("  Rank: %d eval example count:%d" %
                        (hvd.rank(), eval_examples_count))
        tf.logging.info("  Rank: %d eval batch size:%d" %
                        (hvd.rank(), FLAGS.eval_batch_size))
        tf.logging.info("  Rank: %d eval_steps:%d" %
                        (hvd.rank(), eval_steps))

        eval_input_fn = eval_file_based_input_fn_builder(
            input_file=eval_file_list,
            query_seq_length=FLAGS.max_seq_length_query,
            meta_seq_length=meta_seq_length,
            drop_remainder=False,
            is_fidelity_eval=False)

        # create InMemoryEvaluatorHook 
        in_memory_evaluator = tf.estimator.experimental.InMemoryEvaluatorHook(
            estimator=estimator,
            steps=eval_steps, # steps must be set or will not print any log, do not know why
            input_fn=eval_input_fn,
            every_n_iter=FLAGS.save_checkpoints_steps,
            name="fidelity_eval")
        training_hooks.append(in_memory_evaluator)

        train_start_time = time.time()
        estimator.train(input_fn=train_input_fn,
                        max_steps=num_train_steps, hooks=training_hooks)
        train_time_elapsed = time.time() - train_start_time
        train_time_wo_overhead = log_train_run_hook.total_time
        avg_sentences_per_second = num_train_steps * \
            global_batch_size * 1.0 / train_time_elapsed
        ss_sentences_per_second = (
            num_train_steps - log_train_run_hook.skipped) * global_batch_size * 1.0 / train_time_wo_overhead

        if master_process:
            tf.logging.info("-----------------------------")
            tf.logging.info("Total Training Time = %0.2f for Sentences = %d", train_time_elapsed,
                            num_train_steps * global_batch_size)
            tf.logging.info("Total Training Time W/O Overhead = %0.2f for Sentences = %d", train_time_wo_overhead,
                            (num_train_steps - log_train_run_hook.skipped) * global_batch_size)
            tf.logging.info(
                "Throughput Average (sentences/sec) with overhead = %0.2f", avg_sentences_per_second)
            tf.logging.info(
                "Throughput Average (sentences/sec) = %0.2f", ss_sentences_per_second)
            tf.logging.info("-----------------------------")


if __name__ == "__main__":
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("do_train")
    flags.mark_flag_as_required("preprocess_train_dir")
    flags.mark_flag_as_required("train_line_count")
    flags.mark_flag_as_required("train_partition_count")
    flags.mark_flag_as_required("preprocess_train_file_name")
    flags.mark_flag_as_required("preprocess_eval_dir")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("query_bert_config_file")
    flags.mark_flag_as_required("meta_bert_config_file")

    tf.app.run()
