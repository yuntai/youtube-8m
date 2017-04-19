# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains model definitions."""
import math

import models
import tensorflow as tf
import utils

from tensorflow import flags
import tensorflow.contrib.slim as slim

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "moe_hidden_size", 1024,
    "The size of hidden state of MoeModel.")
class DeepMoeModel(models.BaseModel):
  """
    Extension of MoeModel based on
    Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts
    Layer(https://arxiv.org/abs/1701.06538)
  """
  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   hidden_size=None,
                   l2_penalty=1e-5,
                   is_training=True,
                   **unused_params):
    """Creates a Mixture of (Logistic) Experts model.

     The model consists of a per-class softmax distribution over a
     configurable number of logistic classifiers. One of the classifiers in the
     mixture is not trained, and always predicts 0.

    Args:
      model_input: 'batch_size' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
      num_mixtures: The number of mixtures (excluding a dummy 'expert' that
        always predicts the non-existence of an entity).
      l2_penalty: How much to penalize the squared magnitudes of parameter
        values.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
    num_mixtures = num_mixtures or FLAGS.moe_num_mixtures
    hidden_size = hidden_size or FLAGS.moe_hidden_size

    print('model_input:',model_input)
    input_bn = slim.batch_norm(
      model_input,
      center=True,
      scale=True,
      is_training=is_training,
      scope="input_bn"
    )
    print('input_bn:',input_bn)

    gate_hidden = slim.fully_connected(
      input_bn,
      hidden_size,
      activation_fn=None,
      biases_initializer=None,
      weights_regularizer=slim.l2_regularizer(l2_penalty),
      scope="gate_hidden")

    gate_hidden_bn = slim.batch_norm(
      gate_hidden,
      center=True,
      scale=True,
      activation_fn=tf.nn.relu,
      is_training=is_training,
      scope="gates_hidden_bn"
    )
    print("gate_hidden:",gate_hidden)
    print("gate_hidden_bn:",gate_hidden_bn)

    gate_activations = slim.fully_connected(
        gate_hidden_bn,
        vocab_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates")

    expert_hidden = slim.fully_connected(
      input_bn,
      hidden_size,
      activation_fn=None,
      weights_regularizer=slim.l2_regularizer(l2_penalty),
      biases_initializer=None,
      scope="experts_hidden")

    experts_hidden_bn = slim.batch_norm(
      expert_hidden,
      center=True,
      scale=True,
      activation_fn=tf.nn.relu,
      is_training=is_training,
      scope="gates_hidden_bn"
    )

    expert_activations = slim.fully_connected(
        expert_hidden,
        vocab_size * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        biases_initializer=None,
        scope="experts")

    gating_distribution = tf.nn.softmax(tf.reshape(
        gate_activations,
        [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    expert_distribution = tf.nn.sigmoid(tf.reshape(
        expert_activations,
        [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

    final_probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                     [-1, vocab_size])
    return {"predictions": final_probabilities}

class SparseMoeModel(models.BaseModel):
  """
    Extension of MoeModel based on
    Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts
    Layer(https://arxiv.org/abs/1701.06538)
  """
  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-5,
                   is_training=True,
                   **unused_params):
    """Creates a Mixture of (Logistic) Experts model.

     The model consists of a per-class softmax distribution over a
     configurable number of logistic classifiers. One of the classifiers in the
     mixture is not trained, and always predicts 0.

    Args:
      model_input: 'batch_size' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
      num_mixtures: The number of mixtures (excluding a dummy 'expert' that
        always predicts the non-existence of an entity).
      l2_penalty: How much to penalize the squared magnitudes of parameter
        values.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
    num_mixtures = num_mixtures or FLAGS.moe_num_mixtures

    #input_bn = slim.batch_norm(
    #  model_input,
    #  center=True,
    #  scale=True,
    #  is_training=is_training,
    #  scope="input_bn"
    #)
    input_bn = model_input

    gate_activations = slim.fully_connected(
        input_bn,
        num_mixtures+1,
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates")


    noise = tf.random_normal([num_mixtures+1]) * slim.fully_connected(
      input_bn,
      num_mixtures+1,
      activation_fn=tf.nn.softplus
    )

    noise = tf.identity(noise, name="noise")

    noised_gate_activations = tf.add(gate_activations, noise, name="gdadd")

    #v, i = tf.nn.top_k(-noised_gate_activations, k=3)
    #v = tf.expand_dims(v,-1)
    #i = tf.expand_dims(i,-1)
    #print(v)
    #print(i)
    #tf.scatter_nd(i, v, tf.constant([1,10,1]))

    gating_distribution = tf.nn.softmax(noised_gate_activations, name="gd")
    gating_distribution = tf.reshape(gating_distribution, [-1, num_mixtures+1, 1])

    expert_activations = slim.fully_connected(
        input_bn,
        vocab_size * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts")
    expert_distribution = tf.nn.sigmoid(tf.reshape(expert_activations, [-1,
                                                                        num_mixtures,
                                                                        vocab_size]),
                                       name="ed")

    final_probabilities_by_class_and_batch = tf.reduce_sum(
      gating_distribution[:,:num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.identity(final_probabilities_by_class_and_batch,name="fp")

    print('model_input:',model_input)
    print('input_bn:',input_bn)
    print("gate_activations=",gate_activations)
    print("expert_activations=",expert_activations)
    print("gating_distribution:",gating_distribution)
    print("expert_distribution:",expert_distribution)
    print("final_probabilities_by_class_and_batch:",final_probabilities_by_class_and_batch)
    print("final_probabilities:",final_probabilities)

    return {"predictions": final_probabilities}
