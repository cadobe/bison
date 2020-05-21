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
"""bison finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import sys
import os
import tokenization
import tensorflow as tf
from tensorflow.python.client import device_lib
import horovod.tensorflow as hvd
import codecs
import hashlib

csv.field_size_limit(sys.maxsize)
flags = tf.flags

FLAGS = flags.FLAGS

# Required parameters
flags.DEFINE_bool("combine_full_word_and_token_idf", False, "The flag determines if we use token idf when using full word idf")
flags.DEFINE_bool("use_full_word_idf", True, "The flag determines if we use token idf when using full word idf")

flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "query_bison_config_file", None,
    "The config json file corresponding to the pre-trained bison model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "meta_bison_config_file", None,
    "The config json file corresponding to the pre-trained bison model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the bison model was trained on.")
flags.DEFINE_string("full_word_idf_file", None,
                    "The full word idf file that the bison model was trained on.")
flags.DEFINE_string("idf_file", None,
                    "The word piece idf file that the bison model was trained on.")
flags.DEFINE_string(
    "preprocess_train_file", None,
    "The preprocess training file.")

flags.DEFINE_string(
    "preprocess_eval_file", None,
    "The preprocess training file.")

flags.DEFINE_string(
    "preprocess_predict_file", None,
    "The preprocess training file.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

# Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained bison model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length_query", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")
flags.DEFINE_float("default_idf",22.0,"The default value for words not in idf dict.")
flags.DEFINE_integer(
    "max_seq_length_title", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "max_seq_length_anchor", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "max_seq_length_url", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "max_seq_length_click", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_predict", False,
                  "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

# flags.DEFINE_integer("default_idf", 22, "Default value for idf generation.")

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

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_integer("compressor_dim", 100, "output vector size.")
flags.DEFINE_float("nce_temperature", 10, "nce_temperature")
flags.DEFINE_float("nce_weight", 0.5, "nce_weight")
flags.DEFINE_string("activation", 'relu', "activation")
flags.DEFINE_integer("src_col", 0, "src_col")
flags.DEFINE_integer("title_col", 1, "title_col")
flags.DEFINE_integer("anchor_col", 2, "anchor_col")
flags.DEFINE_integer("url_col", 3, "url_col")
flags.DEFINE_integer("click_col", 4, "click_col")
flags.DEFINE_integer("label_col", -1, "label_col")
flags.DEFINE_integer("query_id_col", 6, "query_id_col")
flags.DEFINE_integer("ifm_col", -1, "ifm_col")
flags.DEFINE_integer("instance_id_col", -1, "instance_id_col")
flags.DEFINE_integer("map_label_col", 7, "map_label_col")
flags.DEFINE_integer("docid_col", 8, "docid_col")
flags.DEFINE_string(
    "train_file", "AUTC_QnA_Passage_150MUnique.tsv", "Training file name.")
flags.DEFINE_string(
    "eval_file", "AUTC_QnA_Passage_Evaluation10k.tsv", "Evaluation file name.")
flags.DEFINE_string(
    "predict_file", "AUTCPassageTier012FY19H1DT.tsv", "Predict file name.")
flags.DEFINE_integer("train_line_count", 136490622, "output vector size.")


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, query, title, anchor, url, click, label=None):
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
        self.title = title
        self.anchor = anchor
        self.url = url
        self.click = click
        self.label = label

class SingleFeatureInputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, query, title, anchor, url, click, query_id, ifm, instance_id, map_label, docid, label=None):
        self.guid = guid
        self.query = query
        self.title = title
        self.anchor = anchor
        self.url = url
        self.click = click
        self.label = label
        self.query_id = query_id
        self.ifm = ifm
        self.instance_id = instance_id
        self.map_label = map_label
        self.docid = docid


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, query_input_ids, query_input_mask, query_segment_ids, meta_input_ids, meta_input_mask, metaStream_segment_ids, label_id):
        self.query_input_ids = query_input_ids
        self.query_input_mask = query_input_mask
        self.query_segment_ids = query_segment_ids
        self.meta_input_ids = meta_input_ids
        self.meta_input_mask = meta_input_mask
        self.metaStream_segment_ids = metaStream_segment_ids
        self.label_id = label_id

class SingleFeatureInputFeatures(object):
    """A single set of features of data."""

    def __init__(self, query_input_ids, query_input_mask, query_input_idfs, query_segment_ids, meta_input_ids, meta_input_mask, meta_input_idfs, metaStream_segment_ids, label_id, query_id, ifm, instance_id, map_label, docid):
        self.query_input_ids = query_input_ids
        self.query_input_mask = query_input_mask
        self.query_input_idfs = query_input_idfs
        self.query_segment_ids = query_segment_ids
        self.meta_input_ids = meta_input_ids
        self.meta_input_mask = meta_input_mask
        self.meta_input_idfs = meta_input_idfs
        self.metaStream_segment_ids = metaStream_segment_ids
        self.label_id = label_id
        self.query_id = query_id
        self.ifm = ifm
        self.instance_id = instance_id
        self.map_label = map_label
        self.docid = docid

class SpaceVProcessor(object):
  """Processor for the MRPC data set (GLUE version)."""

  def get_label_map(self):
    """See base class."""
    return {"0": 0, "1": 1, "2": 2}

  def get_rating_map(self):
    """See base class."""
    return {"perfect": 0, "excellent": 1, "good": 2, "fair": 3, "bad": 4}


def convert_single_example(ex_index, example, label_map, rating_map, max_seq_length_query, max_seq_length_title, max_seq_length_anchor, max_seq_length_url, max_seq_length_click,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    query_tokens_a = tokenizer.tokenize(example.query)

    metaStream_title = tokenizer.tokenize(example.title)
    metaStream_anchor = tokenizer.tokenize(example.anchor)
    metaStream_url = tokenizer.tokenize(example.url)
    metaStream_click = tokenizer.tokenize(example.click)

    # Account for [CLS] and [SEP] with "- 2"
    if len(query_tokens_a) > max_seq_length_query - 2:
        query_tokens_a = query_tokens_a[0:(max_seq_length_query - 2)]
    if len(metaStream_title) > max_seq_length_title - 2:
        metaStream_title = metaStream_title[0:(max_seq_length_title - 2)]
    if len(metaStream_anchor) > max_seq_length_anchor - 1:
        metaStream_anchor = metaStream_anchor[0:(max_seq_length_anchor - 1)]
    if len(metaStream_url) > max_seq_length_url - 1:
        metaStream_url = metaStream_url[0:(max_seq_length_url - 1)]
    if len(metaStream_click) > max_seq_length_click - 1:
        metaStream_click = metaStream_click[0:(max_seq_length_click - 1)]

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

    for m_token in metaStream_title:
        metaStream_tokens.append(m_token)
        metaStream_segment_ids.append(0)
    metaStream_tokens.append("[SEP]")
    metaStream_segment_ids.append(0)

    for m_token in metaStream_anchor:
        metaStream_tokens.append(m_token)
        metaStream_segment_ids.append(1)
    metaStream_tokens.append("[SEP]")
    metaStream_segment_ids.append(1)

    for m_token in metaStream_url:
        metaStream_tokens.append(m_token)
        metaStream_segment_ids.append(2)
    metaStream_tokens.append("[SEP]")
    metaStream_segment_ids.append(2)

    for m_token in metaStream_click:
        metaStream_tokens.append(m_token)
        metaStream_segment_ids.append(3)
    metaStream_tokens.append("[SEP]")
    metaStream_segment_ids.append(3)

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
    metaStream_words.extend(basic_tokenizer.tokenize(example.title))
    metaStream_words.append("[SEP]")
    metaStream_words.extend(basic_tokenizer.tokenize(example.anchor))
    metaStream_words.append("[SEP]")
    metaStream_words.extend(basic_tokenizer.tokenize(example.url))
    metaStream_words.append("[SEP]")
    metaStream_words.extend(basic_tokenizer.tokenize(example.click))
    metaStream_words.append("[SEP]")

    if FLAGS.use_full_word_idf:
        if FLAGS.combine_full_word_and_token_idf:
            query_input_idfs = tokenizer.convert_tokens_to_idfs_by_full_word_and_wordpiece(query_tokens, query_words)
            meta_input_idfs = tokenizer.convert_tokens_to_idfs_by_full_word_and_wordpiece(metaStream_tokens, metaStream_words)            
        else:
            # tf.logging.info("Full word IDF is triggered to parse IDF values for token")
            query_input_idfs = tokenizer.convert_tokens_to_bm25_by_full_word(query_tokens, query_words,query_segment_ids, FLAGS.default_idf)
            meta_input_idfs = tokenizer.convert_tokens_to_bm25_by_full_word(metaStream_tokens, metaStream_words,metaStream_segment_ids ,FLAGS.default_idf)
    else:
        # tf.logging.info("Token IDF is triggered to parse IDF values for token")
        query_input_idfs = tokenizer.convert_tokens_to_idfs(query_tokens)
        meta_input_idfs = tokenizer.convert_tokens_to_idfs(metaStream_tokens)

    # Zero-pad up to the sequence length.
    while len(query_input_ids) < max_seq_length_query:
        query_input_ids.append(0)
        query_input_mask.append(0)
        query_segment_ids.append(0)
        query_input_idfs.append(0)
    documents_max_len = max_seq_length_title + max_seq_length_anchor+max_seq_length_url+max_seq_length_click
    while len(meta_input_ids) < documents_max_len:
        meta_input_ids.append(0)
        meta_input_mask.append(0)
        metaStream_segment_ids.append(4)
        meta_input_idfs.append(0)
    
    assert len(query_input_ids) == max_seq_length_query
    assert len(query_input_mask) == max_seq_length_query
    assert len(query_segment_ids) == max_seq_length_query
    assert len(query_input_idfs) == max_seq_length_query
    assert len(meta_input_ids) == documents_max_len
    assert len(meta_input_mask) == documents_max_len
    assert len(metaStream_segment_ids) == documents_max_len
    assert len(meta_input_idfs) == documents_max_len
    
    #FIXME
    # label_id = label_map[example.label]
    if example.map_label.lower() not in rating_map or example.map_label.lower() == 'bad':
        label_id = 0
    else:
        label_id = 1
    #FIXME should balance the single feature data and normal data
    if example.map_label.lower() not in rating_map:
        map_label = rating_map['bad']
    else:
        map_label = rating_map[example.map_label.lower()]
    docid = int(hashlib.sha256(example.docid.encode('utf-8')).hexdigest(), 16) % 10**8

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("query_tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in query_tokens]))
        tf.logging.info("query_words: %s" % " ".join(
            [tokenization.printable_text(x) for x in query_words]))
        tf.logging.info("query_input_ids: %s" %
                        " ".join([str(x) for x in query_input_ids]))
        tf.logging.info("query_input_mask: %s" %
                        " ".join([str(x) for x in query_input_mask]))
        tf.logging.info("query_segment_ids: %s" %
                        " ".join([str(x) for x in query_segment_ids]))
        tf.logging.info("query_input_idfs: %s" % " ".join([str(x) for x in query_input_idfs]))                        
        tf.logging.info("metaStream_tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in metaStream_tokens]))
        tf.logging.info("metaStream_words: %s" % " ".join(
            [tokenization.printable_text(x) for x in metaStream_words]))            
        tf.logging.info("meta_input_ids: %s" %
                        " ".join([str(x) for x in meta_input_ids]))
        tf.logging.info("meta_input_mask: %s" %
                        " ".join([str(x) for x in meta_input_mask]))
        tf.logging.info("metaStream_segment_ids: %s" %
                        " ".join([str(x) for x in metaStream_segment_ids]))
        tf.logging.info("meta_input_idfs: %s" % " ".join([str(x) for x in meta_input_idfs]))                           
        tf.logging.info("label: %s (id = %d)" % (example.label, label_id))
        tf.logging.info("query_id: %s" % (example.query_id))
        tf.logging.info("ifm: %s" % (example.ifm))
        tf.logging.info("instance_id: %s" % (example.instance_id))
        tf.logging.info("map_label: %s (id = %d)" % (example.map_label, map_label))
        tf.logging.info("docid: %s" % (docid))

    feature = SingleFeatureInputFeatures(
        query_input_ids=query_input_ids,
        query_input_mask=query_input_mask,
        query_input_idfs=query_input_idfs,
        query_segment_ids=query_segment_ids,
        meta_input_ids=meta_input_ids,
        meta_input_mask=meta_input_mask,
        meta_input_idfs=meta_input_idfs,        
        metaStream_segment_ids=metaStream_segment_ids,
        label_id=label_id,
        query_id=int(example.query_id),
        ifm=int(example.ifm),
        instance_id=int(example.instance_id),
        map_label=map_label,
        docid=docid)
    return feature

def file_based_convert_examples_to_features_v2(raw_file_path,
                                               csv_line_count,
                                               rank_size,
                                               rank,
                                               label_map,
                                               rating_map,
                                               max_seq_length_query,
                                               max_seq_length_title,
                                               max_seq_length_anchor,
                                               max_seq_length_url,
                                               max_seq_length_click,
                                               tokenizer,
                                               set_type,
                                               src_col,
                                               title_col,
                                               anchor_col,
                                               url_col,
                                               click_col,
                                               label_col,
                                               query_id_col,
                                               ifm_col,
                                               instance_id_col,
                                               map_label_col,
                                               docid_col,
                                               output_file):

    process_count = int(csv_line_count // rank_size)
    process_offset = int(process_count * rank)

    if rank == rank_size - 1:
        process_count += csv_line_count % rank_size

    idx = 0

    with tf.gfile.Open(raw_file_path, 'r') as fp:
        output_file_parent_path = os.path.dirname(output_file)

        if not os.path.exists(output_file_parent_path):
            os.makedirs(output_file_parent_path)

        if os.path.exists(output_file):
            os.remove(output_file)

        tf.logging.info("rank:%d, begin write file to:%s" %
                        (rank, output_file))

        reader = csv.reader((x.replace('\0', '')
                             for x in fp), delimiter="\t", quotechar=None)
        writer = tf.python_io.TFRecordWriter(output_file)

        for line in reader:
            if idx < process_offset:
                if 0 == idx % 1000000:
                    tf.logging.info("rank:%d, skim example %d of %d" %
                                    (rank, idx, process_offset))

                idx += 1
                continue

            if idx >= (process_offset + process_count):
                break

            cur_idx = idx - process_offset

            if cur_idx % 10000 == 0:
                tf.logging.info("rank:%d, Writing example %d of %d" %
                                (rank, cur_idx, process_count))

            # create a single example
            guid = "%s-%s" % (set_type, idx)
            query = tokenization.convert_to_unicode(line[src_col])
            title = tokenization.convert_to_unicode(line[title_col])
            anchor = tokenization.convert_to_unicode(line[anchor_col])
            url = tokenization.convert_to_unicode(line[url_col])
            click = tokenization.convert_to_unicode(line[click_col])
            label = tokenization.convert_to_unicode('1')
            if label_col != -1:
                label = tokenization.convert_to_unicode(line[label_col])
            query_id = tokenization.convert_to_unicode(line[query_id_col])
            ifm = tokenization.convert_to_unicode('1')
            if ifm_col != -1:
                ifm = tokenization.convert_to_unicode(line[ifm_col])
            instance_id = tokenization.convert_to_unicode('1')
            if instance_id_col != -1:
                instance_id = tokenization.convert_to_unicode(line[instance_id_col])
            map_label = tokenization.convert_to_unicode(line[map_label_col])
            docid = tokenization.convert_to_unicode(line[docid_col])

            single_example = SingleFeatureInputExample(guid=guid, query=query, title=title, anchor=anchor, url=url, click=click, 
            query_id=query_id, ifm=ifm, instance_id=instance_id, map_label=map_label, docid=docid, label=label)

            feature = convert_single_example(cur_idx,
                                             single_example,
                                             label_map,
                                             rating_map,
                                             max_seq_length_query,
                                             max_seq_length_title,
                                             max_seq_length_anchor,
                                             max_seq_length_url,
                                             max_seq_length_click,
                                             tokenizer)

            def create_int_feature(values):
                f = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=list(values)))
                return f
            def create_float_feature(values):
                f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
                return f
            features = collections.OrderedDict()
            features["query_input_ids"] = create_int_feature(
                feature.query_input_ids)
            features["query_input_mask"] = create_int_feature(
                feature.query_input_mask)
            features["query_input_idfs"] = create_float_feature(
                feature.query_input_idfs)                
            features["query_segment_ids"] = create_int_feature(
                feature.query_segment_ids)
            features["meta_input_ids"] = create_int_feature(
                feature.meta_input_ids)
            features["meta_input_mask"] = create_int_feature(
                feature.meta_input_mask)
            features["meta_input_idfs"] = create_float_feature(
                feature.meta_input_idfs)                
            features["metaStream_segment_ids"] = create_int_feature(
                feature.metaStream_segment_ids)
            features["label_ids"] = create_int_feature([feature.label_id])
            features["query_id"] = create_int_feature([feature.query_id])
            features["IFM"] = create_int_feature([feature.ifm])
            features["InstanceId"] = create_int_feature([feature.instance_id])
            features["MapLabel"] = create_int_feature([feature.map_label])
            features["docId"] = create_int_feature([feature.docid])

            tf_example = tf.train.Example(
                features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())

            idx += 1

        writer.close()


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    hvd.init()

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    all_meta_seq_length = FLAGS.max_seq_length_title+FLAGS.max_seq_length_anchor + \
        FLAGS.max_seq_length_url+FLAGS.max_seq_length_click

    tf.gfile.MakeDirs(FLAGS.output_dir)

    processor = SpaceVProcessor()

    label_map = processor.get_label_map()

    rating_map = processor.get_rating_map()

    tokenizer = tokenization.FullTokenizerFullWordIDF(
        vocab_file=FLAGS.vocab_file, idf_file=FLAGS.idf_file, full_word_idf_file = FLAGS.full_word_idf_file,  do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    '''read the GPU device info, default is use all GPU, if not set the gpu_device_count paramter'''
    gpu_device_count = 4

    run_config = None

    '''use the flag do different config'''

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    train_examples_count = FLAGS.train_line_count

    if FLAGS.do_train:
        num_train_steps = int(
            train_examples_count / (FLAGS.train_batch_size * gpu_device_count) * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    if FLAGS.do_train:
        if FLAGS.preprocess_train_file is None:
            tf.logging.info("No Preprocessed Training data")
            train_file = os.path.join(
                FLAGS.output_dir, str(hvd.rank()), "train.tf_record")

            raw_train_file_path = os.path.join(
                FLAGS.data_dir, FLAGS.train_file)

            file_based_convert_examples_to_features_v2(raw_train_file_path,
                                                       train_examples_count,
                                                       hvd.size(),
                                                       hvd.rank(),
                                                       label_map,
                                                       rating_map,
                                                       FLAGS.max_seq_length_query,
                                                       FLAGS.max_seq_length_title,
                                                       FLAGS.max_seq_length_anchor,
                                                       FLAGS.max_seq_length_url,
                                                       FLAGS.max_seq_length_click,
                                                       tokenizer,
                                                       "train",
                                                       FLAGS.src_col,
                                                       FLAGS.title_col,
                                                       FLAGS.anchor_col,
                                                       FLAGS.url_col,
                                                       FLAGS.click_col,
                                                       FLAGS.label_col,
                                                       FLAGS.query_id_col,
                                                       FLAGS.ifm_col,
                                                       FLAGS.instance_id_col,
                                                       FLAGS.map_label_col,
                                                       FLAGS.docid_col,
                                                       train_file)
        else:
            train_file = FLAGS.preprocess_train_file
        tf.logging.info("***** Running training *****")

if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("query_bison_config_file")
    flags.mark_flag_as_required("meta_bison_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
