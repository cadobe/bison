# coding=utf-8
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

import collections
import csv
import sys
import os
import tokenization_msmarco_doc as tokenization
import tensorflow as tf
from tensorflow.python.client import device_lib
import horovod.tensorflow as hvd
import codecs
csv.field_size_limit(sys.maxsize)
flags = tf.flags
FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string("task_name", None, "The name of the task.")
flags.DEFINE_string("data_dir", None, "The input data dir. Should contain the .tsv files (or other data files) for the task.")
flags.DEFINE_string("data_file", None, "Data file name.")
flags.DEFINE_integer("data_line_count", None, "Data file line count.")
flags.DEFINE_string("output_dir", None, "The output directory where the preprocessed data will be written.")

flags.DEFINE_string("vocab_file", None, "The vocabulary file.")
flags.DEFINE_string("full_word_idf_file", None, "The idf file that used on full word.")
flags.DEFINE_float("default_idf", None, "The default value for words not in idf dict.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

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

flags.DEFINE_integer("src_col", 2, "src_col")
flags.DEFINE_integer("url_col", 4, "url_col")
flags.DEFINE_integer("title_col", 5, "title_col")
flags.DEFINE_integer("body_col", -1, "body_col")
flags.DEFINE_integer("label_col", -1, "label_col")

flags.DEFINE_float("BM25_K1", 0.25, "K1 in BM25.")
flags.DEFINE_float("BM25_B", 0.4, "B in BM25.")
flags.DEFINE_float("BM25_AVGDL_Q", 6, "AVG Query Len in BM25.")
flags.DEFINE_float("BM25_AVGDL_D", 25, "AVG Doc Len in BM25.")

class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, query, url, title, label=None, body=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.query = query
    self.url = url
    self.title = title
    self.label = label
    self.body = body


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, query_input_ids, query_input_mask,query_input_idfs, query_segment_ids, meta_input_ids, meta_input_mask, meta_input_idfs, metaStream_segment_ids, label_id):
    self.query_input_ids = query_input_ids
    self.query_input_mask = query_input_mask
    self.query_input_idfs = query_input_idfs
    self.query_segment_ids = query_segment_ids
    self.meta_input_ids = meta_input_ids
    self.meta_input_idfs = meta_input_idfs
    self.meta_input_mask = meta_input_mask
    self.metaStream_segment_ids = metaStream_segment_ids
    self.label_id = label_id


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines


class BISONProcessor(DataProcessor):
  """Processor for the MRPC data set (GLUE version)."""
  
  def get_labels(self):
    """See base class."""
    return ["0", "1"]


def convert_single_example(ex_index, example, label_list, 
  max_seq_length_query, max_seq_length_url, max_seq_length_title, max_seq_length_body, tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""
  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  query_tokens_a = tokenizer.tokenize(example.query)

  metaStream_url = tokenizer.tokenize(example.url)
  metaStream_title = tokenizer.tokenize(example.title)
  if example.body is not None:
    metaStream_body = tokenizer.tokenize(example.body)

  # query format: [CLS]query tokens[SEP]
  if len(query_tokens_a) > max_seq_length_query - 2:
    query_tokens_a = query_tokens_a[0:(max_seq_length_query - 2)]
  # document format: [CLS]url tokens[SEP]title tokens[SEP]body tokens(optional)[SEP]
  if len(metaStream_url) > max_seq_length_url - 2:
    metaStream_url = metaStream_url[0:(max_seq_length_url - 2)]
  if len(metaStream_title) > max_seq_length_title - 1:
    metaStream_title = metaStream_title[0:(max_seq_length_title - 1)]
  if example.body is not None:
    if len(metaStream_body) > max_seq_length_body - 1:
      metaStream_body = metaStream_body[0:(max_seq_length_body - 1)]

  query_tokens = []
  metaStream_tokens = []
  query_segment_ids = []
  metaStream_segment_ids = []

  query_tokens.append("[CLS]")
  metaStream_tokens.append("[CLS]")
  query_segment_ids.append(0)
  metaStream_segment_ids.append(0)

  for q_token in query_tokens_a:
    query_tokens.append(q_token)
    query_segment_ids.append(0)
  query_tokens.append("[SEP]")
  query_segment_ids.append(0)

  for m_token in metaStream_url:
    metaStream_tokens.append(m_token)
    metaStream_segment_ids.append(0)
  metaStream_tokens.append("[SEP]")
  metaStream_segment_ids.append(0)

  for m_token in metaStream_title:
    metaStream_tokens.append(m_token)
    metaStream_segment_ids.append(1)
  metaStream_tokens.append("[SEP]")
  metaStream_segment_ids.append(1)

  if example.body is not None:
    for m_token in metaStream_body:
      metaStream_tokens.append(m_token)
      metaStream_segment_ids.append(2)
    metaStream_tokens.append("[SEP]")
    metaStream_segment_ids.append(2)

  query_input_ids = tokenizer.convert_tokens_to_ids(query_tokens)
  meta_input_ids = tokenizer.convert_tokens_to_ids(metaStream_tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  query_input_mask = [1] * len(query_input_ids)
  meta_input_mask = [1] * len(meta_input_ids)

  basic_tokenizer = tokenization.BasicTokenizer(do_lower_case=True)
  query_words = ["[CLS]"]
  query_words.extend(basic_tokenizer.tokenize(example.query))
  query_words.append("[SEP]")

  metaStream_words = ["[CLS]"]
  metaStream_words.extend(basic_tokenizer.tokenize(example.url))
  metaStream_words.append("[SEP]")
  metaStream_words.extend(basic_tokenizer.tokenize(example.title))
  metaStream_words.append("[SEP]")
  if example.body is not None:
    metaStream_words.extend(basic_tokenizer.tokenize(example.body))
    metaStream_words.append("[SEP]")

  query_input_idfs = tokenizer.convert_tokens_to_bm25s_by_full_word_for_msmarco_doc(query_tokens, query_words, FLAGS.default_idf, False)
  meta_input_idfs = tokenizer.convert_tokens_to_bm25s_by_full_word_for_msmarco_doc(metaStream_tokens, metaStream_words, FLAGS.default_idf, True)

  # Zero-pad up to the sequence length.
  while len(query_input_ids) < max_seq_length_query:
    query_input_ids.append(0)
    query_input_mask.append(0)
    query_segment_ids.append(0)
    query_input_idfs.append(0)

  max_seq_length_doc = max_seq_length_url + max_seq_length_title
  if example.body is not None:
    max_seq_length_doc += max_seq_length_body
  while len(meta_input_ids) < max_seq_length_doc:
    meta_input_ids.append(0)
    meta_input_mask.append(0)
    metaStream_segment_ids.append(0)
    meta_input_idfs.append(0)

  assert len(query_input_ids) == max_seq_length_query
  assert len(query_input_mask) == max_seq_length_query
  assert len(query_segment_ids) == max_seq_length_query
  assert len(query_input_idfs) == max_seq_length_query 
  assert len(meta_input_ids) == max_seq_length_doc
  assert len(meta_input_mask) == max_seq_length_doc
  assert len(metaStream_segment_ids) == max_seq_length_doc
  assert len(meta_input_idfs) == max_seq_length_doc

  label_id = label_map[example.label]
  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("query_tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in query_tokens]))
    tf.logging.info("query_input_ids: %s" % " ".join([str(x) for x in query_input_ids]))
    tf.logging.info("query_input_mask: %s" % " ".join([str(x) for x in query_input_mask]))
    tf.logging.info("query_segment_ids: %s" % " ".join([str(x) for x in query_segment_ids]))
    tf.logging.info("query_input_idfs: %s" % " ".join([str(x) for x in query_input_idfs]))
    tf.logging.info("metaStream_tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in metaStream_tokens]))
    tf.logging.info("meta_input_ids: %s" % " ".join([str(x) for x in meta_input_ids]))
    tf.logging.info("meta_input_mask: %s" % " ".join([str(x) for x in meta_input_mask]))
    tf.logging.info("metaStream_segment_ids: %s" % " ".join([str(x) for x in metaStream_segment_ids]))
    tf.logging.info("meta_input_idfs: %s" % " ".join([str(x) for x in meta_input_idfs]))    
    tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

  feature = InputFeatures(
      query_input_ids=query_input_ids,
      query_input_mask=query_input_mask,
      query_input_idfs=query_input_idfs,
      query_segment_ids=query_segment_ids,
      meta_input_ids=meta_input_ids,
      meta_input_mask=meta_input_mask,
      meta_input_idfs=meta_input_idfs,
      metaStream_segment_ids=metaStream_segment_ids,
      label_id=label_id)
  return feature

def file_based_convert_examples_to_features_v2(raw_file_path,
                                               csv_line_count,
                                               rank_size,
                                               rank,
                                               label_list,
                                               max_seq_length_query,
                                               max_seq_length_url,
                                               max_seq_length_title,
                                               max_seq_length_body,
                                               tokenizer,
                                               set_type,
                                               src_col,
                                               url_col,
                                               title_col,
                                               body_col,
                                               label_col,
                                               output_file):
    process_count = int(csv_line_count // rank_size)
    process_offset = int(process_count * rank)

    if rank == rank_size - 1:
        process_count += csv_line_count % rank_size

    idx = 0

    # read the csv file line by line
    with tf.gfile.Open(raw_file_path, 'r') as fp:
        output_file_parent_path = os.path.dirname(output_file)

        if not os.path.exists(output_file_parent_path):
            os.makedirs(output_file_parent_path)

        if os.path.exists(output_file):
            os.remove(output_file)

        tf.logging.info("rank:%d, begin write file to:%s" % (rank, output_file))

        reader = csv.reader((x.replace('\0', '') for x in fp), delimiter="\t", quotechar=None)
        writer = tf.python_io.TFRecordWriter(output_file)

        for line in reader:
            if idx < process_offset:
                if 0 == idx % 1000000:
                    tf.logging.info("rank:%d, skim example %d of %d" % (rank, idx, process_offset))

                idx += 1
                continue

            if idx >= (process_offset + process_count):
                break

            cur_idx = idx - process_offset

            if cur_idx % 10000 == 0:
                tf.logging.info("rank:%d, Writing example %d of %d" % (rank, cur_idx, process_count))

            # create a single example
            guid = "%s-%s" % (set_type, idx)
            query = tokenization.convert_to_unicode(line[src_col])
            url = tokenization.convert_to_unicode(line[url_col])
            title = tokenization.convert_to_unicode(line[title_col])
            label = tokenization.convert_to_unicode('1')
            if not label_col == -1:
              label = tokenization.convert_to_unicode(line[label_col])
            body = None
            if not body_col == -1:
              body = tokenization.convert_to_unicode(line[body_col])

            single_example = InputExample(
              guid=guid, query=query, url=url, title=title, label=label, body=body)

            feature = convert_single_example(cur_idx,
                                             single_example,
                                             label_list,
                                             max_seq_length_query,
                                             max_seq_length_url,
                                             max_seq_length_title,
                                             max_seq_length_body,
                                             tokenizer)

            def create_int_feature(values):
                f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
                return f
            def create_float_feature(values):
                f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
                return f

            features = collections.OrderedDict()
            features["query_input_ids"] = create_int_feature(feature.query_input_ids)
            features["query_input_mask"] = create_int_feature(feature.query_input_mask)
            features["query_input_idfs"] = create_float_feature(feature.query_input_idfs)
            features["query_segment_ids"] = create_int_feature(feature.query_segment_ids)
            features["meta_input_ids"] = create_int_feature(feature.meta_input_ids)
            features["meta_input_mask"] = create_int_feature(feature.meta_input_mask)
            features["meta_input_idfs"] = create_float_feature(feature.meta_input_idfs)
            features["metaStream_segment_ids"] = create_int_feature(feature.metaStream_segment_ids)
            features["label_ids"] = create_int_feature([feature.label_id])

            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())

            idx += 1

        writer.close()
def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  
  tf.logging.info("***** Start to preprocess data *****")
  hvd.init()
  tf.gfile.MakeDirs(FLAGS.output_dir)
  data_file = os.path.join(FLAGS.output_dir, str(hvd.rank()), "train.tf_record")
  raw_data_file_path = os.path.join(FLAGS.data_dir, FLAGS.data_file)
  processors = {
      "bison": BISONProcessor,
  }
  task_name = FLAGS.task_name.lower()
  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))
  processor = processors[task_name]()
  label_list = processor.get_labels()
  tokenizer = tokenization.FullTokenizerFullWordIDF(
        vocab_file=FLAGS.vocab_file, full_word_idf_file=FLAGS.full_word_idf_file,  do_lower_case=FLAGS.do_lower_case) 
  file_based_convert_examples_to_features_v2(raw_data_file_path,
                                            FLAGS.data_line_count,
                                            hvd.size(),
                                            hvd.rank(),
                                            label_list,
                                            FLAGS.max_seq_length_query,
                                            FLAGS.max_seq_length_url,
                                            FLAGS.max_seq_length_title,
                                            FLAGS.max_seq_length_body,
                                            tokenizer,
                                            "train",
                                            FLAGS.src_col,
                                            FLAGS.url_col,
                                            FLAGS.title_col,
                                            FLAGS.body_col,
                                            FLAGS.label_col,
                                            data_file)
  tf.logging.info("***** Data preprocess finished *****")

if __name__ == "__main__":
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("data_file")
  flags.mark_flag_as_required("data_line_count")
  flags.mark_flag_as_required("output_dir")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("full_word_idf_file")
  flags.mark_flag_as_required("default_idf")
  tf.app.run()
