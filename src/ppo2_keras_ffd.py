import math
import numpy as np
import tensorflow as tf
import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import tflearn
import tensorflow.contrib.keras as keras
from keras.layers import Input, Dense, concatenate, Lambda
from keras.layers.core import Flatten
from keras.layers.convolutional import Conv1D
from keras.models import Model

import global_constants as settings

# FEATURE_NUM = 64 #128
ACTION_EPS = 1e-6
GAMMA = 0.99
# PPO2
EPS = 0.2

class Network():
    def CreateNetwork(self, inputs, n_dense=1):
        with tf.variable_scope('actor'):
            assert n_dense >= 0

            # Concatenate inputs into 1 vector
            # input_vector = []
            # input_vector.append(inputs[:, 0:1, -1])
            # input_vector.append(inputs[:, 1:2, -1])
            # input_vector.append(tf.squeeze(inputs[:, 2:3, :], axis=1))
            # input_vector.append(tf.squeeze(inputs[:, 3:4, :], axis=1))
            # input_vector.append(tf.squeeze(inputs[:, 4:5, :self.a_dim], axis=1))
            # input_vector.append(inputs[:, 5:6, -1])
            # input_vector = concatenate(input_vector, axis=-1)
            # keras_inputs = Input(tensor=input_vector)
            print(len(inputs.shape))
            keras_inputs = Input(tensor=inputs)
            # layer = input_vector
            layer = keras_inputs
            for i in range(n_dense):
                layer = Dense(settings.FEATURE_NUM,
                    activation='relu',
                    kernel_initializer='truncated_normal')(layer)

            base_dense = Dense(settings.FEATURE_NUM,
                activation='relu',
                kernel_initializer='truncated_normal')(layer)

            # Original Architecture
            net = Dense(settings.FEATURE_NUM,
                activation='relu',
                kernel_initializer='truncated_normal')(base_dense)


            pi = Dense(self.a_dim,
                activation='relu',
                kernel_initializer='truncated_normal')(net)
            value = Dense(1,
                activation='linear',
                kernel_initializer='truncated_normal')(net)

            outputs = concatenate([pi, value], axis=-1)

            # model = Model(inputs=keras_inputs, outputs=[pi, value])
            model = Model(inputs=keras_inputs, outputs=value)

            return pi, value, model
            # value = Dense(1,
            #     activation='linear',
            #     kernel_initializer='truncated_normal')(net)
            # model = Model(inputs=keras_inputs, outputs=value)
            # return value, value, model

    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, input_network_params):
        # print("input network params: {}".format(input_network_params))
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)
        })

    def __init__(self, sess, state_dim, action_dim, learning_rate, n_dense=1):
        self.quality = 0
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr_rate = learning_rate
        self.sess = sess
        self.R = tf.placeholder(tf.float32, [None, 1])
        # self.inputs = tf.placeholder(tf.float32, [None, self.s_dim[0], self.s_dim[1]])
        self.inputs = tf.placeholder(tf.float32, [None, 25])
        self.old_pi = tf.placeholder(tf.float32, [None, self.a_dim])
        self.acts = tf.placeholder(tf.float32, [None, self.a_dim])
        self.entropy_weight = tf.placeholder(tf.float32)
        self.pi, self.val, self.model = self.CreateNetwork(inputs=self.inputs, n_dense=n_dense)
        self.real_out = tf.clip_by_value(self.pi, ACTION_EPS, 1. - ACTION_EPS)
        self.entropy = tf.multiply(self.real_out, tf.log(self.real_out))
        self.adv = tf.stop_gradient(self.R - self.val)
        self.ratio = tf.reduce_sum(tf.multiply(self.real_out, self.acts), reduction_indices=1, keep_dims=True) / \
                tf.reduce_sum(tf.multiply(self.old_pi, self.acts), reduction_indices=1, keep_dims=True)

        self.ppo2loss = tf.minimum(self.ratio * self.adv,
                            tf.clip_by_value(self.ratio, 1 - EPS, 1 + EPS) * self.adv
                        )

        # Get all network parameters
        self.network_params = \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')

        # Set all network parameters
        self.input_network_params = []

        for param in self.network_params:
            self.input_network_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        self.set_network_params_op = []
        for idx, param in enumerate(self.input_network_params):
            self.set_network_params_op.append(
                self.network_params[idx].assign(param))

        self.loss = tflearn.mean_square(self.val, self.R) \
            - tf.reduce_sum(self.ppo2loss) \
            + self.entropy_weight * tf.reduce_sum(self.entropy)

        self.optimize = tf.train.AdamOptimizer(self.lr_rate).minimize(self.loss)

    def predict(self, input):
        # reshape inputs if necessary
        if len(input.shape) == 3:
            input_vector = []
            input_vector.append(input[:, 0:1, -1])
            input_vector.append(input[:, 1:2, -1])
            input_vector.append(tf.squeeze(input[:, 2:3, :], axis=1))
            input_vector.append(tf.squeeze(input[:, 3:4, :], axis=1))
            input_vector.append(tf.squeeze(input[:, 4:5, :self.a_dim], axis=1))
            input_vector.append(input[:, 5:6, -1])
            input_vector = concatenate(input_vector, axis=-1)
            input_vector = input
        action = self.sess.run(self.real_out, feed_dict={
            self.inputs: input
        })
        return action[0]

    def get_entropy(self, step):
        if step < 20000:
            return 5.
        elif step < 50000:
            return 1.
        elif step < 70000:
            return 0.7
        elif step < 90000:
            return 0.5
        elif step < 120000:
            return 0.3
        else:
            return 0.1

    def train(self, s_batch, a_batch, p_batch, v_batch, epoch, batch_size = 128):
        # shuffle is all you need
        s_batch, a_batch, p_batch, v_batch = \
            tflearn.data_utils.shuffle(s_batch, a_batch, p_batch, v_batch)
        # mini_batch
        i, train_len = 0, s_batch.shape[0]
        while train_len >= 0:
            self.sess.run(self.optimize, feed_dict={
                self.inputs: s_batch[i:i+batch_size],
                self.acts: a_batch[i:i+batch_size],
                self.R: v_batch[i:i+batch_size],
                self.old_pi: p_batch[i:i+batch_size],
                self.entropy_weight: self.get_entropy(epoch)
            })
            train_len -= batch_size
            i += batch_size

    def compute_v(self, s_batch, a_batch, r_batch, terminal):
        ba_size = len(s_batch)
        R_batch = np.zeros([len(r_batch), 1])

        if terminal:
            R_batch[-1, 0] = 0  # terminal state
        else:
            v_batch = self.sess.run(self.val, feed_dict={
                self.inputs: s_batch
            })
            R_batch[-1, 0] = v_batch[-1, 0]  # boot strap from last state
        for t in reversed(range(ba_size - 1)):
            R_batch[t, 0] = r_batch[t] + GAMMA * R_batch[t + 1, 0]

        return list(R_batch)
