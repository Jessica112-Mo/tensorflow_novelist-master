# Copyright 2017 Jin Fagang. All Rights Reserved.
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
# =======================================================================

#  RNN 模型层

import tensorflow as tf
from tensorflow.contrib import rnn


class RNNModel(object):
    """
     @:param
      inputs :  输入
      labels :  标签
      n_units :
      n_layers :
      lr :
      vocab_size : 词典规模
    """
    def __init__(self, inputs, labels, n_units, n_layers, lr, vocab_size):
        self.inputs = inputs
        self.labels = labels
        self.n_units = n_units
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.lr = lr

        self.outputs, self.states = self.rnn_model()

        if self.labels is not None:
            self.train_op, self.loss = self.update()

    def rnn_model(self):
        # BasicLSTMCell 最基本的LSTM循环网络单元，添加forget_bias(默认值是1)到遗忘门的偏置。为了减少在开始训练时遗忘的规模，他
        # 不允许单元有一个裁剪，映射层不允许有peep-hole连接，这是基准。
        # BasicLSTMCell 的实现类在 rnn.python.ops下， core_rnn_cell_impl.py
        cell = rnn.BasicLSTMCell(num_units=self.n_units)

        # MultiRNNCell 这个函数有两个参数：第一个参数就是输入的RNN的实例形成的列表，第二个参数就是让状态是
        # 一个元组，官方推荐是True    state_is_tuple = True
        # 可以实现多层的LSTM网络，将前一层的输出作为后一层的输入
        multi_cell = rnn.MultiRNNCell([cell]*self.n_layers)

        # we only need one output so get it wrapped to out one value which is next word index
        # 将 rnn_cell 的输出映射成想要的维度 output_size是映射后的size 返回一个带Output_projection的rnn_cell
        cell_wrapped = rnn.OutputProjectionWrapper(multi_cell, output_size=1)

        # get input embed
        # tf.random_uniform(shape, minval, maxval, dtype, seed, name) ： 返回一个 n*n的矩阵，值产生于minval 和 maxval 之间
        embedding = tf.Variable(initial_value=tf.random_uniform([self.vocab_size, self.n_units], -1.0, 1.0))

        # tf.nn.embedding_lokkup(embedding, inputs_id) : 根据inputs_id寻找embedding中对应的元素。比如，input_ids=[1,3,5],则
        # 找出embedding中下标为1,3,5的向量组成一个矩阵返回。
        inputs = tf.nn.embedding_lookup(embedding, self.inputs)
        # what is inputs dim??

        # add initial state into dynamic rnn, if I am not result would be bad, I tried, don't know why
        if self.labels is not None:

        # zero_state ； 参数初始化
            initial_state = cell_wrapped.zero_state(int(inputs.get_shape()[0]), tf.float32)
        else:
            initial_state = cell_wrapped.zero_state(1, tf.float32)

        # dynamic_rnn 实现的功能可以让不同迭代的batch是不同长度的数据，但同一次迭代一个batch内部的所有数据长度仍然是固定的。
        # dynamic_rnn 和 rnn 比较
        outputs, states = tf.nn.dynamic_rnn(cell_wrapped, inputs=inputs, dtype=tf.float32, initial_state=initial_state)
        outputs = tf.reshape(outputs, [int(outputs.get_shape()[0]), int(inputs.get_shape()[1])])

        # truncated_normal : 截断分布，详见高斯分布
        w = tf.Variable(tf.truncated_normal([int(inputs.get_shape()[1]), self.vocab_size]))
        b = tf.Variable(tf.zeros([self.vocab_size]))

        logits = tf.nn.bias_add(tf.matmul(outputs, w), b)
        return logits, states

    def update(self):
        # one_hot:
        labels_one_hot = tf.one_hot(tf.reshape(self.labels, [-1]), depth=self.vocab_size)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels_one_hot, logits=self.outputs)
        total_loss = tf.reduce_mean(loss)
        # 详见tf优化器博客和tf的APi
        train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss=loss)
        return train_op, total_loss








