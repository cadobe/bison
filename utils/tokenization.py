#!/usr/bin/env python # -*- coding: utf-8 -*-
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
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

"""Tokenization classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import unicodedata
import six
import tensorflow as tf
import re
import os

METASTREAMWEIGHT = {'A':51,'U':46,'T':46,'B':1}
# AverageMetaStreamLength ={'A':682,'U': 7 /* 7.034 */,'T':6 /* 6.368 */,'B':6829}
AverageMetaStreamLength ={'A':682,'U':7,'T':6,'B':6829}
MetaStreamLengthNorm={'A':0.60,'U':0.86,'T':0.86,'B':0.94}


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


def printable_text(text):
    """Returns text encoded in a way suitable for print or `tf.logging`."""

    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with tf.gfile.GFile(vocab_file, "r") as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab

def load_idf(idf_file):
    """"Loads a idf file that containing all words' idf value"""
    idf = collections.OrderedDict()
    with tf.gfile.GFile(idf_file, "r") as reader:
        while True:
            line = convert_to_unicode(reader.readline())
            if not line:
                break
            line_arr = line.strip().split("\t")
            if len(line_arr) < 2:
                continue
            token = line_arr[0]
            idf_instance = float(line_arr[1])
            idf[token] = idf_instance
    return idf

def convert_token_to_idfs(idfs, tokens):
    """Convert a series of tokenId to its idf"""
    idfs_list = []
    for token in tokens:
        idfs_list.append(idfs[token])
    return idfs_list

def convert_tokens_to_tfidfs_by_full_word(full_word_idfs, tokens, words,default_idf):
    import collections
    tfs = collections.Counter(words)
    tfidfs_list = []
    n_words = len(words)
    token_idx = 0
    word_idx = 0
    temp_token_agg = ""
    accumulation = 1
    while token_idx < len(tokens) and word_idx < len(words):
        temp_token_agg += tokens[token_idx] if "##" not in tokens[token_idx] else tokens[token_idx][2:]
        while tokens[token_idx] == "[SEP]" and words[word_idx] != "[SEP]":
            word_idx += 1
        word = words[word_idx]
        if temp_token_agg == word or (token_idx < len(tokens) - 1 and tokens[token_idx + 1] == "[SEP]"):
            idf = full_word_idfs[word] if word in full_word_idfs else default_idf
            for i in range(accumulation):
                tfidfs_list.append(idf*tfs[word]/n_words)
            temp_token_agg = ""
            accumulation = 1
            word_idx += 1
            token_idx += 1
        else:
            accumulation += 1
            token_idx += 1
    return tfidfs_list

def adjusted_term_frequency(term_frequency,metastream_length,metastream_type):
    metastream_weight = METASTREAMWEIGHT[metastream_type]
    metastream_length_norm = MetaStreamLengthNorm[metastream_type]
    metastream_avg_len = AverageMetaStreamLength[metastream_type]
    atf= metastream_weight * term_frequency /(1.0+metastream_length_norm * \
    (metastream_length/metastream_avg_len -1.0))
    return atf

def bm25(atf,idf):
    K1 = 37
    return (atf/(K1+atf))*idf

def convert_tokens_to_bm25_by_full_word(full_word_idfs, tokens, words, segment_ids, default_idf):
    import collections
    tfs = dict()
    atf_title,atf_anchor,atf_url,atf_click = dict(),dict(),dict(),dict()
    bm25_list = []
    sep_indices = [i for i, x in enumerate(words) if x == "[SEP]"]
    is_query = False
    if len(sep_indices) == 1:
        is_query =True
        tfs = collections.Counter(words)
    else:
        assert len(sep_indices) == 4
        words_title = words[:sep_indices[0]+1]
        words_anchor = words[sep_indices[0]+1:sep_indices[1]+1]
        words_url= words[sep_indices[1]+1:sep_indices[2]+1]
        words_click= words[sep_indices[2]+1:]
        tf_title = collections.Counter(words_title)
        tf_anchor = collections.Counter(words_anchor)
        tf_url = collections.Counter(words_url)
        tf_click = collections.Counter(words_click)
        for word,tf in tf_title.items():
            atf = adjusted_term_frequency(tf,len(words_title),'T')
            atf_title[word] = atf
        for word,tf in tf_anchor.items():
            atf = adjusted_term_frequency(tf,len(words_anchor),'A')
            atf_anchor[word] = atf            
        for word,tf in tf_url.items():
            atf = adjusted_term_frequency(tf,len(words_url),'U')
            atf_url[word] = atf              
        for word,tf in tf_click.items():
            atf = adjusted_term_frequency(tf,len(words_click),'T')
            atf_click[word] = atf          
    n_words = len(words)
    token_idx = 0
    word_idx = 0
    temp_token_agg = ""
    accumulation = 1
    while token_idx < len(tokens) and word_idx < len(words):
        temp_token_agg += tokens[token_idx] if "##" not in tokens[token_idx] else tokens[token_idx][2:]
        while tokens[token_idx] == "[SEP]" and words[word_idx] != "[SEP]":
            word_idx += 1
        word = words[word_idx]
        if temp_token_agg == word or (token_idx < len(tokens) - 1 and tokens[token_idx + 1] == "[SEP]"):
            idf = full_word_idfs[word] if word in full_word_idfs else default_idf
            for i in range(accumulation):
                if is_query:
                    bm25_list.append(idf*tfs[word]/n_words)
                else:
                    assert len(set(segment_ids)) == 4
                    if segment_ids[token_idx] == 0:
                        bm25_list.append(bm25(atf_title[word],idf) if bm25(atf_title[word],idf) >0 else 1 ) 
                    elif segment_ids[token_idx] ==1:
                        bm25_list.append(bm25(atf_anchor[word],idf) if bm25(atf_anchor[word],idf)>0 else 1)
                    elif segment_ids[token_idx] == 2:
                        bm25_list.append(bm25(atf_url[word],idf) if bm25(atf_url[word],idf)>0 else 1)
                    else:
                        bm25_list.append(bm25(atf_click[word],idf) if bm25(atf_click[word],idf)>0 else 1)
            temp_token_agg = ""
            accumulation = 1
            word_idx += 1
            token_idx += 1
        else:
            accumulation += 1
            token_idx += 1
    return bm25_list

def convert_tokens_to_tfidfs_by_full_word(full_word_idfs, tokens, words,default_idf):
    import collections
    tfs = collections.Counter(words)
    tfidfs_list = []
    n_words = len(words)
    token_idx = 0
    word_idx = 0
    temp_token_agg = ""
    accumulation = 1
    while token_idx < len(tokens) and word_idx < len(words):
        temp_token_agg += tokens[token_idx] if "##" not in tokens[token_idx] else tokens[token_idx][2:]
        while tokens[token_idx] == "[SEP]" and words[word_idx] != "[SEP]":
            word_idx += 1
        word = words[word_idx]
        if temp_token_agg == word or (token_idx < len(tokens) - 1 and tokens[token_idx + 1] == "[SEP]"):
            idf = full_word_idfs[word] if word in full_word_idfs else default_idf
            for i in range(accumulation):
                tfidfs_list.append(idf*tfs[word]/n_words)
            temp_token_agg = ""
            accumulation = 1
            word_idx += 1
            token_idx += 1
        else:
            accumulation += 1
            token_idx += 1
    return tfidfs_list

#For tokens originated from one same word, mark their IDF as the same with the Word IDF
def convert_tokens_to_idfs_by_full_word(full_word_idfs, tokens, words, default_idf):
    idfs_list = []
    token_idx = 0
    word_idx = 0
    temp_token_agg = ""
    accumulation = 1
    while token_idx < len(tokens) and word_idx < len(words):
        temp_token_agg += tokens[token_idx] if "##" not in tokens[token_idx] else tokens[token_idx][2:]
        while tokens[token_idx] == "[SEP]" and words[word_idx] != "[SEP]":
            word_idx += 1
        word = words[word_idx]
        if temp_token_agg == word or (token_idx < len(tokens) - 1 and tokens[token_idx + 1] == "[SEP]"):
            idf = full_word_idfs[word] if word in full_word_idfs else default_idf
            for i in range(accumulation):
                idfs_list.append(idf)
            temp_token_agg = ""
            accumulation = 1
            word_idx += 1
            token_idx += 1
        else:
            accumulation += 1
            token_idx += 1
    return idfs_list

def convert_tokens_to_idfs_by_full_word_avged_len(full_word_idfs, tokens, words, default_idf):
    idfs_list = []
    token_idx = 0
    word_idx = 0
    temp_token_agg = ""
    accumulation = 1
    while token_idx < len(tokens) and word_idx < len(words):
        temp_token_agg += tokens[token_idx] if "##" not in tokens[token_idx] else tokens[token_idx][2:]
        while tokens[token_idx] == "[SEP]" and words[word_idx] != "[SEP]":
            word_idx += 1
        word = words[word_idx]
        if temp_token_agg == word or (token_idx < len(tokens) - 1 and tokens[token_idx + 1] == "[SEP]"):
            idf = full_word_idfs[word] if word in full_word_idfs else default_idf
            for i in range(accumulation):
                idfs_list.append(idf/accumulation)
            temp_token_agg = ""
            accumulation = 1
            word_idx += 1
            token_idx += 1
        else:
            accumulation += 1
            token_idx += 1
    return idfs_list    

#For tokens originated from one same word, mark their IDF as the same with the Word IDF, for those not covered in the word idf file, use its wordpiece idf as replacement
def convert_tokens_to_idfs_by_full_word_and_wordpiece(full_word_idfs,token_idfs, tokens, words):
    idfs_list = []
    token_idx = 0
    word_idx = 0
    temp_token_agg = ""
    temp_token_list = list()
    accumulation = 1
    while token_idx < len(tokens) and word_idx < len(words):
        cur_token = tokens[token_idx] if "##" not in tokens[token_idx] else tokens[token_idx][2:]
        temp_token_agg += cur_token
        temp_token_list.append(cur_token)
        while tokens[token_idx] == "[SEP]" and words[word_idx] != "[SEP]":
            word_idx += 1
        word = words[word_idx]
        if temp_token_agg == word or (token_idx < len(tokens) - 1 and tokens[token_idx + 1] == "[SEP]"):
            if word in full_word_idfs:
                idf = full_word_idfs[word]
                for i in range(accumulation):
                    idfs_list.append(idf)                
            else:
                for temp_token in temp_token_list:
                    idfs_list.append(token_idfs[temp_token])
            temp_token_agg = ""
            temp_token_list.clear()
            accumulation = 1
            word_idx += 1
            token_idx += 1
        else:
            accumulation += 1
            token_idx += 1
    return idfs_list

def convert_by_vocab(vocab, items):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for item in items:
        output.append(vocab[item])
    return output


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a peice of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens
    
class FullTokenizerFullWordIDF(object):
    # Run a tokenization init with IDF information
    def __init__(self, vocab_file,idf_file,full_word_idf_file, do_lower_case=True):
        self.vocab = load_vocab(vocab_file)
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
        self.token_idfs = load_idf(idf_file)
        self.full_word_idfs = load_idf(full_word_idf_file)
    
    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
        
        return split_tokens
    
    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.vocab, tokens)
    
    def convert_tokens_to_idfs_by_full_word(self, tokens, words, default_idf):
        return convert_tokens_to_idfs_by_full_word(self.full_word_idfs, tokens, words,default_idf)

    def convert_tokens_to_tfidfs_by_full_word(self, tokens, words, default_idf):
        return convert_tokens_to_tfidfs_by_full_word(self.full_word_idfs, tokens, words,default_idf)

    def convert_tokens_to_bm25_by_full_word(self, tokens, words, segment_ids, default_idf):
        return convert_tokens_to_bm25_by_full_word(self.full_word_idfs, tokens, words,segment_ids,default_idf)        

    def convert_tokens_to_idfs_by_full_word_avged_len(self, tokens, words, default_idf):
        return convert_tokens_to_idfs_by_full_word_avged_len(self.full_word_idfs, tokens, words,default_idf)

    def convert_tokens_to_idfs_by_full_word_and_wordpiece(self,tokens,words):
        return convert_tokens_to_idfs_by_full_word_and_wordpiece(self.full_word_idfs,self.token_idfs,tokens,words)
    
class FullTokenizer(object):
    """Runs end-to-end tokenziation."""

    def __init__(self, vocab_file, do_lower_case=True):
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)

        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.inv_vocab, ids)


class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, do_lower_case=True):
        """Constructs a BasicTokenizer.

    Args:
      do_lower_case: Whether to lower case the input.
    """
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = convert_to_unicode(text)
        text = self._clean_text(text)
        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.

    This uses a greedy longest-match-first algorithm to perform tokenization
    using the given vocabulary.

    For example:
      input = "unaffable"
      output = ["un", "##aff", "##able"]

    Args:
      text: A single token or whitespace separated tokens. This should have
        already been passed through `BasicTokenizer.

    Returns:
      A list of wordpiece tokens.
    """

        text = convert_to_unicode(text)

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False
