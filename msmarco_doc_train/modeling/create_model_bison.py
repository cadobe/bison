import tensorflow as tf
from modeling import modeling_bison, optimization_nvidia

flags = tf.flags
FLAGS = flags.FLAGS

def file_based_input_fn_builder(input_file, batch_size, query_seq_length, meta_seq_length, is_training,
                                drop_remainder, is_fidelity_eval=False, hvd=None):
    """Creates an `input_fn` closure to be passed to Estimator."""

    name_to_features = {
        "query_input_ids": tf.FixedLenFeature([query_seq_length], tf.int64),
        "query_input_mask": tf.FixedLenFeature([query_seq_length], tf.int64),
        "query_input_idfs": tf.FixedLenFeature([query_seq_length], tf.float32),
        "query_segment_ids": tf.FixedLenFeature([query_seq_length], tf.int64),
        "meta_input_ids": tf.FixedLenFeature([meta_seq_length], tf.int64),
        "meta_input_mask": tf.FixedLenFeature([meta_seq_length], tf.int64),
        "meta_input_idfs": tf.FixedLenFeature([meta_seq_length], tf.float32),
        "metaStream_segment_ids": tf.FixedLenFeature([meta_seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
    }
    name_to_features_eval = {
        "query_input_ids": tf.FixedLenFeature([query_seq_length], tf.int64),
        "query_input_mask": tf.FixedLenFeature([query_seq_length], tf.int64),
        "query_input_idfs": tf.FixedLenFeature([query_seq_length], tf.float32),
        "query_segment_ids": tf.FixedLenFeature([query_seq_length], tf.int64),
        "meta_input_ids": tf.FixedLenFeature([meta_seq_length], tf.int64),
        "meta_input_mask": tf.FixedLenFeature([meta_seq_length], tf.int64),
        "meta_input_idfs": tf.FixedLenFeature([meta_seq_length], tf.float32),
        "metaStream_segment_ids": tf.FixedLenFeature([meta_seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
        "query_id": tf.FixedLenFeature([], tf.int64),
        "IFM": tf.FixedLenFeature([], tf.int64),
        "InstanceId": tf.FixedLenFeature([], tf.int64),
        "MapLabel": tf.FixedLenFeature([], tf.int64),
        "docId": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn():
        """The actual input function."""

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        if is_training:
            files = tf.data.Dataset.list_files(input_file)
            d = files.interleave(tf.data.TFRecordDataset, cycle_length=32,
                num_parallel_calls=24)
            d = d.prefetch(buffer_size=batch_size * 4 * 3)
            d = d.shuffle(buffer_size=batch_size * 3)
            d = d.repeat()
            d = d.map(map_func=lambda record: _decode_record(record, name_to_features),
                  num_parallel_calls=24)
            d = d.batch(batch_size) 
        else:
            d = tf.data.TFRecordDataset(input_file)
            d = d.apply(
                tf.contrib.data.map_and_batch(
                    lambda record: _decode_record(record,
                                              name_to_features=name_to_features_eval if is_fidelity_eval else name_to_features),
                    batch_size=batch_size,
                    drop_remainder=drop_remainder))
        return d

    return input_fn


def eval_file_based_input_fn_builder(input_file, query_seq_length, meta_seq_length, drop_remainder=False, is_fidelity_eval=False):
    name_to_features = {
        "query_input_ids": tf.FixedLenFeature([query_seq_length], tf.int64),
        "query_input_mask": tf.FixedLenFeature([query_seq_length], tf.int64),
        "query_input_idfs": tf.FixedLenFeature([query_seq_length], tf.float32),
        "query_segment_ids": tf.FixedLenFeature([query_seq_length], tf.int64),
        "meta_input_ids": tf.FixedLenFeature([meta_seq_length], tf.int64),
        "meta_input_mask": tf.FixedLenFeature([meta_seq_length], tf.int64),
        "meta_input_idfs": tf.FixedLenFeature([meta_seq_length], tf.float32),
        "metaStream_segment_ids": tf.FixedLenFeature([meta_seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
    }

    name_to_features_eval = {
        "query_input_ids": tf.FixedLenFeature([query_seq_length], tf.int64),
        "query_input_mask": tf.FixedLenFeature([query_seq_length], tf.int64),
        "query_input_idfs": tf.FixedLenFeature([query_seq_length], tf.float32),
        "query_segment_ids": tf.FixedLenFeature([query_seq_length], tf.int64),
        "meta_input_ids": tf.FixedLenFeature([meta_seq_length], tf.int64),
        "meta_input_mask": tf.FixedLenFeature([meta_seq_length], tf.int64),
        "meta_input_idfs": tf.FixedLenFeature([meta_seq_length], tf.float32),
        "metaStream_segment_ids": tf.FixedLenFeature([meta_seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
        "query_id": tf.FixedLenFeature([], tf.int64),
        "IFM": tf.FixedLenFeature([], tf.int64),
        "InstanceId": tf.FixedLenFeature([], tf.int64),
        "MapLabel": tf.FixedLenFeature([], tf.int64),
        "docId": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = FLAGS.eval_batch_size

        d = tf.data.TFRecordDataset(input_file)
        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record,
                                              name_to_features=name_to_features_eval if is_fidelity_eval else name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def create_model(query_bert_config, meta_bert_config, is_training, query_input_ids, query_input_mask, query_input_idfs, query_segment_ids, meta_input_ids, meta_input_mask, meta_input_idfs, metaStream_segment_ids,
                 labels, use_one_hot_embeddings, nce_temperature, nce_weight):
    """Creates a classification model."""
    tf.logging.info("*** Query Weighted attention is enabled in Transformer ***")
    query_model = modeling_bison.BertModel(
        config=query_bert_config,
        is_training=is_training,
        input_ids=query_input_ids,
        input_mask=query_input_mask,
        input_idfs=query_input_idfs,
        token_type_ids=query_segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings,
        scope="query",
        compute_type=tf.float16 if FLAGS.use_fp16 else tf.float32)

    tf.logging.info("*** Meta Weighted attention is enabled in Transformer ***")    
    meta_model = modeling_bison.BertModel(
        config=meta_bert_config,
        is_training=is_training,
        input_ids=meta_input_ids,
        input_mask=meta_input_mask,
        input_idfs=meta_input_idfs,
        token_type_ids=metaStream_segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings,
        scope="meta",
        compute_type=tf.float16 if FLAGS.use_fp16 else tf.float32)

    query_output_layer = query_model.get_pooled_output()
    tf.logging.info(
        "  !!!!!!!!!!!!!!!!!!!!!!!!!query_output_layer = %s", query_output_layer)

    meta_output_layer = meta_model.get_pooled_output()
    tf.logging.info(
        "  !!!!!!!!!!!!!!!!!!!!!!!!!meta_output_layer = %s", meta_output_layer)
    query_hidden_size = query_output_layer.shape[-1].value
    meta_hidden_size = meta_output_layer.shape[-1].value

    tf.summary.histogram("query_output_layer", query_output_layer)
    tf.summary.histogram("meta_output_layer", meta_output_layer)

    with tf.variable_scope("loss"):
        query_vectors = query_output_layer
        meta_vectors = meta_output_layer
        tf.logging.info(
            "  !!!!!!!!!!!!!!!!!!!!!!!!!query_vectors = %s", query_vectors)
        tf.logging.info(
            "  !!!!!!!!!!!!!!!!!!!!!!!!!meta_vectors = %s", meta_vectors)

        # Filter NCE cases by query similarity
        query_output_layer_l2 = tf.nn.l2_normalize(query_vectors,-1)
        t_cross_query_sim = tf.matmul(query_output_layer_l2,query_output_layer_l2, transpose_b=True)
        t_cross_query_mask = tf.where(tf.greater(t_cross_query_sim,0.90),-1e12 *tf.ones_like(t_cross_query_sim),tf.zeros_like(t_cross_query_sim)) 

        # Generate NCE cases
        meta_encodesrc = meta_output_layer
        batch_size = tf.shape(meta_encodesrc)[0]
        t_encoded_src_norm = tf.nn.l2_normalize(query_vectors, -1)
        t_encoded_trg_norm = tf.nn.l2_normalize(meta_vectors, -1)
        tf.logging.info(
            "  !!!!!!!!!!!!!!!!!!!!!!!!!t_encoded_src_norm = %s", t_encoded_src_norm)
        tf.logging.info(
            "  !!!!!!!!!!!!!!!!!!!!!!!!!t_encoded_trg_norm = %s", t_encoded_trg_norm)

        t_cross_sim = tf.matmul(
            t_encoded_src_norm, t_encoded_trg_norm, transpose_b=True)
        t_cross_sim_masked = -1e12 * \
            tf.eye(tf.shape(t_cross_sim)[0]) + t_cross_sim
        t_cross_sim_masked = t_cross_query_mask + t_cross_sim_masked 
        tf.logging.info(
            "  !!!!!!!!!!!!!!!!!!!!!!!!!t_cross_sim_masked = %s", t_cross_sim_masked)
        t_max_neg_idx = tf.reshape(tf.multinomial(
            t_cross_sim_masked * nce_temperature, 1), [-1])
        t_max_neg_idx = tf.stop_gradient(t_max_neg_idx)
        tf.logging.info(
            "  !!!!!!!!!!!!!!!!!!!!!!!!!t_max_neg_idx = %s", t_max_neg_idx)
        t_neg_encoded_trg_norm = tf.gather(t_encoded_trg_norm, t_max_neg_idx)
        tf.logging.info(
            "  !!!!!!!!!!!!!!!!!!!!!!!!!t_neg_encoded_trg_norm = %s", t_neg_encoded_trg_norm)

        t_encoded_src_norm_concat = tf.concat(
            [t_encoded_src_norm, t_encoded_src_norm], 0)
        t_encoded_trg_norm_concat = tf.concat(
            [t_encoded_trg_norm, t_neg_encoded_trg_norm], 0)
        tf.logging.info(
            "  !!!!!!!!!!!!!!!!!!!!!!!!!Add_t_encoded_src_norm = %s", t_encoded_src_norm_concat)
        tf.logging.info(
            "  !!!!!!!!!!!!!!!!!!!!!!!!!Add_t_encoded_trg_norm = %s", t_encoded_trg_norm_concat)

        tf.logging.info("  !!!!!!!!!!!!!!!!!!!!!!!!!raw_labels = %s", labels)
        t_label = tf.to_int32(labels)
        t_label = tf.pad(t_label, [[0, tf.shape(labels)[0]]])
        tf.logging.info("  !!!!!!!!!!!!!!!!!!!!!!!!!pad_labels = %s", t_label)

        t_sim = tf.reduce_sum(
            t_encoded_src_norm_concat * t_encoded_trg_norm_concat,
            -1)

        pos_logits=tf.boolean_mask(t_sim[:batch_size],tf.equal(t_label[:batch_size],1)) 
        neg_logits=tf.boolean_mask(t_sim[:batch_size],tf.equal(t_label[:batch_size],0))
        tf.summary.scalar('pos_logits', tf.reduce_mean(pos_logits))
        tf.summary.scalar('neg_logits', tf.reduce_mean(neg_logits)) 
        tf.summary.scalar('nce_logits', tf.reduce_mean(t_sim[batch_size:]))
        if FLAGS.Fix_Sim_Weight:
            t_logits = FLAGS.sim_weight * t_sim + FLAGS.sim_bias
        else:
            v_weights = tf.get_variable(
                'SimWeights',
                initializer=tf.ones([1], dtype=tf.float32))
            v_biases = tf.get_variable(
                'SimBiases',
                initializer=tf.zeros([1], dtype=tf.float32))
            # monitor sim_weight and sim_bias change
            tf.summary.scalar('sim_weight', v_weights[0])
            tf.summary.scalar('sim_bias', v_biases[0])
            t_logits = v_weights * t_sim + v_biases

        loss_label = tf.where(tf.equal(t_label,1),t_label,tf.zeros_like(t_label))
        per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.to_float(loss_label), logits=t_logits)
        tf.logging.info(
            "  !!!!!!!!!!!!!!!!!!!!!!!!!per_example_loss = %s", per_example_loss)
        eval_loss = tf.reduce_mean(per_example_loss[:batch_size], 0)
        tf.logging.info("  !!!!!!!!!!!!!!!!!!!!!!!!!eval_loss = %s", eval_loss)
        nce_loss = tf.reduce_mean(per_example_loss[batch_size:], 0)
        tf.logging.info("  !!!!!!!!!!!!!!!!!!!!!!!!!nce_loss = %s", nce_loss)
        loss = nce_weight * nce_loss + (1.0 - nce_weight) * eval_loss

        tf.summary.scalar('eval_loss', (1.0 - nce_weight) * eval_loss)
        tf.summary.scalar('nce_loss', nce_weight * nce_loss)

        loss = tf.identity(loss, name='loss')
        per_example_loss = tf.identity(
            per_example_loss, name='per_example_loss')
        query_vectors = tf.identity(query_vectors, name='query_vectors')
        meta_vectors = tf.identity(meta_vectors, name='meta_vectors')
        score = tf.identity(t_sim[:batch_size], name='score')
        return (loss, per_example_loss, query_vectors, meta_vectors, score)


def model_fn_builder(query_bert_config, meta_bert_config, init_checkpoint,
                     learning_rate, num_train_steps, num_warmup_steps, use_one_hot_embeddings, nce_temperature, nce_weight, hvd=None):
    """Returns `model_fn` closure for Estimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for Estimator."""
        def metric_fn(per_example_loss, label_ids, cos_distance):
            predictions = tf.where(tf.greater(cos_distance, 0.5), x=tf.ones_like(
                cos_distance), y=tf.zeros_like(cos_distance))
            accuracy = tf.metrics.accuracy(label_ids, tf.sigmoid(cos_distance))
            loss = tf.metrics.mean(per_example_loss)
            return {
                "eval_accuracy": accuracy,
                "eval_loss": loss,
            }

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" %
                            (name, features[name].shape))

        query_input_ids = features["query_input_ids"]
        query_input_mask = features["query_input_mask"]
        query_input_idfs = features["query_input_idfs"]
        query_segment_ids = features["query_segment_ids"]
        meta_input_ids = features["meta_input_ids"]
        meta_input_idfs = features["meta_input_idfs"]
        meta_input_mask = features["meta_input_mask"]
        metaStream_segment_ids = features["metaStream_segment_ids"]
        label_ids = features["label_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        (total_loss, per_example_loss, query_vectors, meta_vectors, cos_distance) = create_model(
            query_bert_config, meta_bert_config, is_training, 
            query_input_ids, query_input_mask,query_input_idfs, query_segment_ids, meta_input_ids, meta_input_mask, meta_input_idfs, metaStream_segment_ids, label_ids,
            use_one_hot_embeddings, nce_temperature, nce_weight)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        if init_checkpoint and (hvd is None or hvd.rank() == 0):
            (assignment_map, initialized_variable_names
             ) = modeling_bison.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        if FLAGS.verbose_logging:
            tf.logging.info("**** Trainable Variables ****")
            for var in tvars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"
                tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                                init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op, LR, l2_norm = optimization_nvidia.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps,
                hvd, False, FLAGS.use_fp16, FLAGS.num_accumulation_steps)
            tf.logging.info("  loss = %s", total_loss)
            tf.summary.scalar('learning_rate', LR)

            logging_hook = tf.estimator.LoggingTensorHook({"loss": total_loss, "learning_rate": LR, "global_norm": l2_norm}, every_n_iter=100)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                training_hooks=[logging_hook])
        elif mode == tf.estimator.ModeKeys.EVAL:
            eval_metrics = (
                metric_fn, [per_example_loss, label_ids, cos_distance])
            
            predictions = tf.where(tf.greater(cos_distance, 0.5), x=tf.ones_like(cos_distance), y=tf.zeros_like(cos_distance))

            accuracy = tf.metrics.accuracy(label_ids, predictions)
            auc = tf.metrics.auc(label_ids, predictions)
            eval_metric_ops_dict = {'accuracy': accuracy,
                                    'auc': auc}
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metric_ops_dict)
        else:
            cos_distance = tf.expand_dims(cos_distance, -1)
            tf.logging.info("  cos_distance = %s, shape = %s",
                            cos_distance.name, cos_distance.shape)
            rt = cos_distance
            tf.logging.info("  rt = %s, shape = %s", rt.name, rt.shape)
            output_spec = tf.estimator.EstimatorSpec(mode=mode, predictions=rt)
        return output_spec

    return model_fn