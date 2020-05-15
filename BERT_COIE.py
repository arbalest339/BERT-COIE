#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Copyright 2018 The Google AI Language Team Authors.
BASED ON Google_BERT.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils.data_processor import InputFeatures, convert_single_example, OreProcessor
from bert import modeling
from bert import optimization
from bert import tokenization
from bert_config import FLAGS
from utils.post_process import emerge
import tensorflow as tf
import numpy as np
from utils import tf_metrics
import pickle
import collections
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def filed_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file, mode=None
):
    label_map = {}
    for (i, label) in enumerate(label_list, 1):  # label_map是标记，标记对应它在表中的序号
        label_map[label] = i
    with open('./output/label2id.pkl', 'wb') as w:
        pickle.dump(label_map, w)
    origin_path = os.path.dirname(os.path.abspath(__file__))
    posfile = os.path.join(origin_path, "data/pos_mat.npy")  # 读取位置嵌入的矩阵
    dptfile = os.path.join(origin_path, "data/dp_mat.npy")  # 读取实体类型的矩阵
    headfile = os.path.join(origin_path, "data/head_mat.npy")
    pos_mat = np.load(posfile)  # 读取两个矩阵
    dp_mat = np.load(dptfile)
    head_mat = np.load(headfile)

    writer = tf.python_io.TFRecordWriter(output_file)
    for (ex_index, example) in enumerate(examples):
        feature = convert_single_example(ex_index, example, pos_mat, dp_mat, head_mat, label_map, max_seq_length,
                                         tokenizer, mode)  # 代码中将每一条数据封装成tfrecord的形式
        pos_feature = feature.pos_embedding.flatten()  # 先将位置数组坍缩
        dp_feature = feature.dp_embedding.flatten()  # 先将位置数组坍缩

        if ex_index % 5000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        features["pos_embedding"] = tf.train.Feature(float_list=tf.train.FloatList(value=pos_feature))
        features["dp_embedding"] = tf.train.Feature(float_list=tf.train.FloatList(value=dp_feature))
        # 此处放入position
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))  # 总体自组装为record
        writer.write(tf_example.SerializeToString())  # 写入特征暂存文件

    if mode == 'test':  # 补足最后一个batch
        batch_size = FLAGS.predict_batch_size
        if not len(examples) % batch_size == 0:
            input_ids = [0] * max_seq_length
            input_mask = [0] * max_seq_length
            segment_ids = [0] * max_seq_length
            label_ids = [0] * max_seq_length
            pos_embedding = np.zeros((max_seq_length, 10), dtype=np.float)
            dp_embedding = np.zeros((max_seq_length, 15), dtype=np.float)
            feature = InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_ids=label_ids,
                pos_embedding=pos_embedding,
                dp_embedding=dp_embedding
                # label_mask = label_mask
            )
            for i in range(batch_size - len(examples) % batch_size):
                features = collections.OrderedDict()
                features["input_ids"] = create_int_feature(feature.input_ids)
                features["input_mask"] = create_int_feature(feature.input_mask)
                features["segment_ids"] = create_int_feature(feature.segment_ids)
                features["label_ids"] = create_int_feature(feature.label_ids)
                features["pos_embedding"] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=pos_feature))  # 此处放入position
                features["dp_embedding"] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=dp_feature))  # 此处放入position
                tf_example = tf.train.Example(features=tf.train.Features(feature=features))  # 总体自组装为record
                writer.write(tf_example.SerializeToString())  # 写入特征暂存文件


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),  # 这里的feature应该是建模时得到的feature
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "pos_embedding": tf.VarLenFeature(tf.float32),  # 这里加入pos_embedding一项
        "dp_embedding": tf.VarLenFeature(tf.float32)  # 这里加入pos_embedding一项
        # "label_ids":tf.VarLenFeature(tf.int64),
        # "label_mask": tf.FixedLenFeature([seq_length], tf.int64),
    }

    def _decode_record(record, name_to_features):  # 把长整型转化为32位整型
        example = tf.parse_single_example(record, name_to_features)  # 处理从文件中读出的数据

        pos_embedding = tf.sparse_tensor_to_dense(example['pos_embedding'], default_value=0)
        pos_embedding = tf.reshape(pos_embedding, [seq_length, 10])
        example['pos_embedding'] = pos_embedding

        dp_embedding = tf.sparse_tensor_to_dense(example['dp_embedding'], default_value=0)
        dp_embedding = tf.reshape(dp_embedding, [seq_length, 15])
        example['dp_embedding'] = dp_embedding

        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):  # 这里是把数据分词batch的地方
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)  # 从tf_record文件中读取数据
        if is_training:
            d = d.repeat()  # 分为epoche
            d = d.shuffle(buffer_size=100)  # 混洗，注意这里和batch无关
        d = d.apply(tf.contrib.data.map_and_batch(  # 与特征名对应后分出batch
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder
        ))
        return d

    return input_fn


def create_model(bert_config, is_training, input_ids, input_mask,
                 segment_ids, labels, pos_embedding, dp_embedding, num_labels,
                 use_one_hot_embeddings):  # 这里是构建模型的重点，需要改变
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )

    output_layer = model.get_sequence_output()  # 计算图：获得bert的输出
    '''
    output_layer : float Tensor of shape [batch_size, seq_length, hidden_size], the final
    hidden layer of the Transformer.
    '''
    output_layer = tf.concat([output_layer, pos_embedding, dp_embedding], -1)
    hidden_size = output_layer.shape[-1].value

    output_weight = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02)
    )
    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer()
    )
    with tf.variable_scope("loss"):
        if is_training:
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
        output_layer = tf.reshape(output_layer, [-1, hidden_size])
        logits = tf.matmul(output_layer, output_weight, transpose_b=True)  # (1024, 788) * (7, 788)^T 维度缩减
        logits = tf.nn.bias_add(logits, output_bias)  # (1024,7) + (7,)
        logits = tf.reshape(logits, [-1, FLAGS.max_seq_length, num_labels])  # 维度还原(8, 128, 7)

        if is_training:
            length = tf.constant(FLAGS.max_seq_length, shape=[FLAGS.train_batch_size, ], dtype=tf.int32)
        else:
            length = tf.constant(FLAGS.max_seq_length, shape=[FLAGS.eval_batch_size, ], dtype=tf.int32)

        # 注意！！！！crf要求每个batch都是足量的，所以使用时要么丢弃最后一个不足量的batch，要么补足
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(  # 得到最大然似和转移矩阵
            inputs=logits,  # 输入的特征向量 [batch_size, max_seq_len, num_tags]
            tag_indices=labels,  # 目标标签 [batch_size, max_seq_len]   #空的转移矩阵
            sequence_lengths=length)  # （batch，）每个序列的长度
        predict, viterbi_score = tf.contrib.crf.crf_decode(logits, transition_params, length)
        loss = tf.reduce_mean(-log_likelihood)

        #
        # log_probs = tf.nn.log_softmax(logits, axis=-1)      #计算对数然似损失，与logit维数相同(8, 128, 7)
        # one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)  #独热向量（7, 7）
        # per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)  #相乘后，第三维求和，得到(8, 128)
        # loss = tf.reduce_sum(per_example_loss)      #整个batch的loss
        # probabilities = tf.nn.softmax(logits, axis=-1)      #计算最大然似损失,得到各标签概率，(8, 128, 7)
        # predict = tf.argmax(probabilities, axis=-1)         #取第3维最大值为预测结果

        return (loss, logits, predict)
        ##########################################################################


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, valid_labels):
    def model_fn(features, labels, mode, params):
        tf.logging.info("*** Features ***")  # logging 用来记录日志
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        pos_embedding = features["pos_embedding"]  # 增加获取位置向量
        dp_embedding = features["dp_embedding"]  # 增加获取位置向量
        # label_mask = features["label_mask"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, logits, predicts) = create_model(  # 使用BERT的接口建模
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids, pos_embedding, dp_embedding,
            num_labels, use_one_hot_embeddings)
        tvars = tf.trainable_variables()  # 得到所有要训练的变量
        scaffold_fn = None
        if init_checkpoint:  # 用BERT预加载模型，这里加载的只有BERT预训练的模型
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                       init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)  # 使用预训练模型
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        tf.logging.info("**** Trainable Variables ****")

        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:  # 这里输出的是预加载模型中的向量格式
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)
        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:  # 在训练阶段
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)  # 创建一个Adam优化器
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(  # TPU运行时的特殊 estimator
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:  # 评估阶段

            def metric_fn(label_ids, predicts, valid_labels):
                # def metric_fn(label_ids, logits):
                # predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)      #直接计算第三维最大值为预测值
                precision = tf_metrics.precision(label_ids, predicts, num_labels, valid_labels,
                                                 average="macro")  # 对比实际值和预测值计算正确率
                recall = tf_metrics.recall(label_ids, predicts, num_labels, valid_labels,
                                           average="macro")  # 对比实际值和预测值计算召回率
                f = tf_metrics.f1(label_ids, predicts, num_labels, valid_labels, average="macro")  # 对比实际值和预测值计算F值
                #
                return {
                    "eval_precision": precision,
                    "eval_recall": recall,
                    "eval_f": f,
                    # "eval_loss": loss,
                }

            eval_metrics = (metric_fn, [label_ids, predicts, valid_labels])
            # eval_metrics = (metric_fn, [per_example_loss, label_ids, logits])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, predictions=predicts, scaffold_fn=scaffold_fn
            )
        return output_spec

    return model_fn


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    processors = {
        "ore": OreProcessor  # added
    }

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    # if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    #     raise ValueError(
    #         "Cannot use sequence length %d because the BERT model "
    #         "was only trained up to sequence length %d" %
    #         (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    task_name = FLAGS.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    processor = processors[task_name]()

    label_list = processor.get_labels()  # labellist里就是所有的label类型
    valid_label = processor.valid_labels()

    tokenizer = tokenization.FullTokenizer(  # 生成一个用基础分词和wordpiece分词的分词器，理论上说我们应该用的是basic分词
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)  # 此处对输入的文件进行了处理结果为list[uid=train+序号，汉字，标签]的形式
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(  # 使用bert的结构和预训练的模型建模
        bert_config=bert_config,
        num_labels=len(label_list) + 1,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,  # 总训练步数
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu,
        valid_labels=valid_label
    )

    estimator = tf.contrib.tpu.TPUEstimator(  # 构造bert的estimator封装器
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")

        filed_based_convert_examples_to_features(  # 同时读取基础向量和位置向量
            train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file, 'train')  # 分词器用于字到向量的转换
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(  # 读取record 数据，组成batch
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)  # 放入数据，执行训练

    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")

        filed_based_convert_examples_to_features(  # 同时读取基础向量和位置向量
            eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file, 'eval')

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d", len(eval_examples))
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
        eval_steps = None
        if FLAGS.use_tpu:
            eval_steps = int(len(eval_examples) / FLAGS.eval_batch_size)
        eval_drop_remainder = True
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)
        result = estimator.evaluate(input_fn=eval_input_fn)
        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        output_predict_file = os.path.join(FLAGS.output_dir, "label_test.txt")

    if FLAGS.do_predict:
        token_path = os.path.join(FLAGS.output_dir, "token_test.txt")
        with open('./output/label2id.pkl', 'rb') as rf:
            label2id = pickle.load(rf)
            id2label = {value: key for key, value in label2id.items()}
        if os.path.exists(token_path):
            os.remove(token_path)
        predict_examples = processor.get_test_examples(FLAGS.data_dir)

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")

        filed_based_convert_examples_to_features(  # 同时读取基础向量和位置向量
            predict_examples, label_list, FLAGS.max_seq_length, tokenizer, predict_file, 'test')

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d", len(predict_examples))
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
        if FLAGS.use_tpu:
            # Warning: According to tpu_estimator.py Prediction on TPU is an
            # experimental feature and hence not supported here
            raise ValueError("Prediction in TPU not supported")
        predict_drop_remainder = False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = estimator.predict(input_fn=predict_input_fn)
        output_predict_file = os.path.join(FLAGS.output_dir, "label_test.txt")

        with open(output_predict_file, 'w') as writer:
            for prediction in result:
                output_line = " ".join(id2label[id] for id in prediction if id != 0) + "\n"
                writer.write(output_line)


if __name__ == "__main__":
    '''
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    '''
    tf.app.run()
