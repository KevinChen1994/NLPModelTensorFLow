# -*- coding:utf-8 -*-
# author: KevinChen1994
# datetime:2023/4/12 16:29
# Description:
import tensorflow as tf
from bert import modeling


class SimCSE(object):
    def __init__(self,
                 bert_config,
                 max_seq_len,
                 learning_rate,
                 is_training):
        self.input_ids = tf.placeholder(dtype=tf.int32, shape=[None, max_seq_len],
                                        name='input_ids')  # batch_size * sequence_len
        self.input_mask = tf.placeholder(dtype=tf.int32, shape=[None, max_seq_len],
                                         name='input_mask')
        self.segment_ids = tf.placeholder(dtype=tf.int32, shape=[None, max_seq_len],
                                          name='segment_ids')

        model = modeling.BertModel(config=bert_config,
                                   is_training=is_training,
                                   input_ids=self.input_ids,
                                   input_mask=self.input_mask,
                                   token_type_ids=self.segment_ids,
                                   use_one_hot_embeddings=False)

        self.bert_output = model.get_pooled_output()
        self.loss = self.simcse_unsup_loss()
        # self.loss = self.simcse_sup_loss()
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    def simcse_unsup_loss(self, temp=0.05):
        batch_size = tf.shape(self.bert_output)[0]
        # 得到y_pred对应的label, [1, 0, 3, 2, ..., batch_size-1, batch_size-2]
        y_true = tf.range(0, batch_size)
        y_true = (y_true - y_true % 2 * 2) + 1
        # L2正则化后再进行内积即可得到余弦相似度，也可以直接通过api进行计算余弦相似度，得到相似度矩阵(对角矩阵)
        norm_emb = tf.math.l2_normalize(self.bert_output, axis=1)
        similarities = tf.matmul(norm_emb, norm_emb, transpose_b=True)
        # 将相似度矩阵对角线置为很小的值, 消除自身的影响
        similarities = similarities - tf.eye(batch_size) * 1e12
        # 相似度矩阵除以温度系数
        similarities = similarities / temp
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=similarities, labels=y_true)
        return tf.reduce_mean(loss)

    def simcse_sup_loss(self, temp=0.05):
        batch_size = tf.shape(self.bert_output)[0]
        row = tf.range(0, batch_size, 3, dtype=tf.int32)
        col = tf.range(0, batch_size, dtype=tf.int32)
        mask = tf.math.not_equal(col % 3, 0)
        col = tf.boolean_mask(col, mask)
        # 生成真实的label
        y_true = tf.range(0, tf.shape(col)[0], 2, dtype=tf.int32)

        # 计算余弦相似度
        norm_emb = tf.math.l2_normalize(self.bert_output, axis=1)
        similarities = tf.matmul(norm_emb, norm_emb, transpose_b=True)
        # 获取对应的相似度矩阵
        similarities = tf.gather(similarities, row, axis=0)
        similarities = tf.gather(similarities, col, axis=1)
        similarities = similarities / temp
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=similarities, labels=y_true)
        return tf.reduce_mean(loss)


if __name__ == '__main__':
    bert_config_path = ''
    bert_ckpt_path = ''

    bert_config = modeling.BertConfig.from_json_file(bert_config_path)
    model = SimCSE(bert_config, 6, 1e-5, True)
    tvars = tf.trainable_variables()
    (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                               bert_ckpt_path)
    tf.train.init_from_checkpoint(bert_ckpt_path, assignment_map)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 随便写的输入数据
        feed_dict = {model.input_ids: [[101, 682, 682, 682, 682, 102], [101, 682, 682, 682, 682, 102],
                                       [101, 672, 672, 672, 672, 172], [101, 672, 672, 672, 672, 102]],
                     model.input_mask: [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]],
                     model.segment_ids: [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1],
                                         [1, 1, 1, 1, 1, 1]]}
        output, loss = sess.run([model.bert_output, model.loss], feed_dict=feed_dict)
