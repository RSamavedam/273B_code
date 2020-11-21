# Copyright 2019 Calico LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
#from __future__ import print_function

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
import sys
import time

from natsort import natsorted
import numpy as np
import tensorflow as tf

from basenji import blocks
from basenji import layers
from basenji import metrics



import random

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import optimizer

GATE_OP = 1
print("in seqnn, we want to check if tensorflow is still executing eagerly")
print(tf.executing_eagerly())

class CustomModel(tf.keras.Model):
    def train_step(self, data):
        #tf.compat.v1.enable_eager_execution()
        print("Important, is tensorflow executing eagerly")
        print(tf.executing_eagerly())
        print("tensorflow version number")
        print(tf.__version__)
        x, y = data 
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            print(type(y_pred))
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        print(loss)
        print(y.shape)
        test_optimizer = PCGrad(self.optimizer)
        gradients = tape.gradient(loss, self.trainable_variables)
        print("gradients calculated")
        print(self.optimizer)
        #print(None in gradients)
        #better_gradients = [0 if i==NoneType else i for i in gradients]
        #print(None in better_gradients)
        #print("calculated better gradients")
        #print(better_gradients)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


class SeqNN():

  def __init__(self, params):
    self.set_defaults()
    for key, value in params.items():
      self.__setattr__(key, value)
    self.build_model()
    self.ensemble = None
    self.embed = None

  def set_defaults(self):
    # only necessary for my bespoke parameters
    # others are best defaulted closer to the source
    self.augment_rc = False
    self.augment_shift = 0

  def build_block(self, current, block_params):
    """Construct a SeqNN block.
    Args:
    Returns:
      current
    """
    block_args = {}

    # extract name
    block_name = block_params['name']
    del block_params['name']

    # if Keras, get block variables names
    pass_all_globals = True
    if block_name[0].isupper():
      pass_all_globals = False
      block_func = blocks.keras_func[block_name]
      block_varnames = block_func.__init__.__code__.co_varnames

    # set global defaults
    global_vars = ['activation', 'batch_norm', 'bn_momentum', 'bn_type',
      'l2_scale', 'l1_scale']
    for gv in global_vars:
      gv_value = getattr(self, gv, False)
      if gv_value and (pass_all_globals or gv in block_varnames):
        block_args[gv] = gv_value

    # set remaining params
    block_args.update(block_params)

    # switch for block
    if block_name[0].islower():
      block_func = blocks.name_func[block_name]
      current = block_func(current, **block_args)

    else:
      block_func = blocks.keras_func[block_name]
      current = block_func(**block_args)(current)

    return current

  def build_model(self, save_reprs=False):
    ###################################################
    # inputs
    ###################################################
    sequence = tf.keras.Input(shape=(self.seq_length, 4), name='sequence')
    # self.genome = tf.keras.Input(shape=(1,), name='genome')
    current = sequence

    # augmentation
    if self.augment_rc:
      current, reverse_bool = layers.StochasticReverseComplement()(current)
    current = layers.StochasticShift(self.augment_shift)(current)

    ###################################################
    # build convolution blocks
    ###################################################
    for bi, block_params in enumerate(self.trunk):
      current = self.build_block(current, block_params)

    # final activation
    current = layers.activate(current, self.activation)

    # make model trunk
    trunk_output = current
    self.model_trunk = tf.keras.Model(inputs=sequence, outputs=trunk_output)

    ###################################################
    # heads
    ###################################################
    self.preds_triu = False

    head_keys = natsorted([v for v in vars(self) if v.startswith('head')])
    self.heads = [getattr(self, hk) for hk in head_keys]

    self.head_output = []
    for hi, head in enumerate(self.heads):
      if not isinstance(head, list):
        head = [head]

      # reset to trunk output
      current = trunk_output

      # build blocks
      for bi, block_params in enumerate(head):
        self.preds_triu |= (block_params['name'] == 'upper_tri')
        current = self.build_block(current, block_params)

      # transform back from reverse complement
      if self.augment_rc:
        if self.preds_triu:
          current = layers.SwitchReverseTriu(self.diagonal_offset)([current, reverse_bool])
        else:
          current = layers.SwitchReverse()([current, reverse_bool])

      # save head output
      self.head_output.append(current)

    ###################################################
    # compile model(s)
    ###################################################
    self.models = []
    for ho in self.head_output:
        self.models.append(tf.keras.Model(inputs=sequence, outputs=ho))
    self.model = self.models[0]
    print(self.model.summary())

    ###################################################
    # track pooling/striding and cropping
    ###################################################
    self.model_strides = []
    self.target_lengths = []
    self.target_crops = []
    for model in self.models:
      self.model_strides.append(1)
      for layer in self.model.layers:
        if hasattr(layer, 'strides'):
          self.model_strides[-1] *= layer.strides[0]
      if type(sequence.shape[1]) == tf.compat.v1.Dimension:
        target_full_length = sequence.shape[1].value // self.model_strides[-1]
      else:
        target_full_length = sequence.shape[1] // self.model_strides[-1]

      self.target_lengths.append(model.outputs[0].shape[1])
      if type(self.target_lengths[-1]) == tf.compat.v1.Dimension:
        self.target_lengths[-1] = self.target_lengths[-1].value
      self.target_crops.append((target_full_length - self.target_lengths[-1])//2)
    print('model_strides', self.model_strides)
    print('target_lengths', self.target_lengths)
    print('target_crops', self.target_crops)


  def build_embed(self, conv_layer_i, batch_norm=True):
    if conv_layer_i == -1:
      self.embed = tf.keras.Model(inputs=self.model.inputs,
                                  outputs=self.model.inputs)
    else:
      if batch_norm:
        conv_layer = self.get_bn_layer(conv_layer_i)
      else:
        conv_layer = self.get_conv_layer(conv_layer_i)

      self.embed = tf.keras.Model(inputs=self.model.inputs,
                                  outputs=conv_layer.output)


  def build_ensemble(self, ensemble_rc=False, ensemble_shifts=[0]):
    """ Build ensemble of models computing on augmented input sequences. """
    if ensemble_rc or len(ensemble_shifts) > 1:
      # sequence input
      sequence = tf.keras.Input(shape=(self.seq_length, 4), name='sequence')
      sequences = [sequence]

      if len(ensemble_shifts) > 1:
        # generate shifted sequences
        sequences = layers.EnsembleShift(ensemble_shifts)(sequences)

      if ensemble_rc:
        # generate reverse complements and indicators
        sequences_rev = layers.EnsembleReverseComplement()(sequences)
      else:
        sequences_rev = [(seq,tf.constant(False)) for seq in sequences]

      # predict each sequence
      if self.preds_triu:
        preds = [layers.SwitchReverseTriu(self.diagonal_offset)
                  ([self.model(seq), rp]) for (seq,rp) in sequences_rev]
      else:
        preds = [layers.SwitchReverse()([self.model(seq), rp]) for (seq,rp) in sequences_rev]

      # create layer
      preds_avg = tf.keras.layers.Average()(preds)      

      # create meta model
      self.ensemble = tf.keras.Model(inputs=sequence, outputs=preds_avg)


  def build_slice(self, target_slice=None):
    if target_slice is not None:
      if len(target_slice) < self.num_targets():
        # sequence input
        sequence = tf.keras.Input(shape=(self.seq_length, 4), name='sequence')

        # predict
        predictions = self.model(sequence)

        # slice
        predictions_slice = tf.gather(predictions, target_slice, axis=-1)

        # replace model
        self.model = tf.keras.Model(inputs=sequence, outputs=predictions_slice)


  def evaluate(self, seq_data, head_i=0, loss='poisson'):
    """ Evaluate model on SeqDataset. """
    # choose model
    if self.ensemble is None:
      model = self.models[head_i]
    else:
      model = self.ensemble

    # compile with dense metrics
    num_targets = self.model.output_shape[-1]

    if loss == 'bce':
      model.compile(optimizer=tf.keras.optimizers.SGD(),
                    loss=loss,
                    metrics=[metrics.SeqAUC(curve='ROC', summarize=False),
                             metrics.SeqAUC(curve='PR', summarize=False)])
    else:      
      model.compile(optimizer=tf.keras.optimizers.SGD(),
                    loss=loss,
                    metrics=[metrics.PearsonR(num_targets, summarize=False),
                             metrics.R2(num_targets, summarize=False)])

    # evaluate
    return model.evaluate(seq_data.dataset)


  def get_bn_layer(self, bn_layer_i=0):
    """ Return specified batch normalization layer. """
    bn_layers = [layer for layer in self.model.layers if layer.name.startswith('batch_normalization')]
    return bn_layers[bn_layer_i]


  def get_conv_layer(self, conv_layer_i=0):
    """ Return specified convolution layer. """
    conv_layers = [layer for layer in self.model.layers if layer.name.startswith('conv')]
    return conv_layers[conv_layer_i]


  def get_conv_weights(self, conv_layer_i=0):
    """ Return kernel weights for specified convolution layer. """
    conv_layer = self.get_conv_layer(conv_layer_i)
    weights = conv_layer.weights[0].numpy()
    weights = np.transpose(weights, [2,1,0])
    return weights


  def num_targets(self, head_i=None):
    if head_i is None:
      return self.model.output_shape[-1]
    else:
      return self.models[head_i].output_shape[-1]


  def predict(self, seq_data, head_i=0, generator=False, **kwargs):
    """ Predict targets for SeqDataset. """
    # choose model
    if self.embed is not None:
      model = self.embed
    elif self.ensemble is not None:
      model = self.ensemble
    else:
      model = self.models[head_i]

    dataset = getattr(seq_data, 'dataset', None)
    if dataset is None:
      dataset = seq_data

    if generator:
      return model.predict_generator(dataset, **kwargs)
    else:
      return model.predict(dataset, **kwargs)


  def restore(self, model_file, trunk=False):
    """ Restore weights from saved model. """
    if trunk:
      self.model_trunk.load_weights(model_file)
    else:
      self.model.load_weights(model_file)


  def save(self, model_file, trunk=False):
    if trunk:
      self.model_trunk.save(model_file)
    else:
      self.model.save(model_file)





import random

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import optimizer

GATE_OP = 1


class PCGrad(optimizer.Optimizer):
    '''Tensorflow implementation of PCGrad.
    Gradient Surgery for Multi-Task Learning: https://arxiv.org/pdf/2001.06782.pdf
    '''

    def __init__(self, optimizer, use_locking=False, name="PCGrad"):
        """optimizer: the optimizer being wrapped
        """
        super(PCGrad, self).__init__(use_locking, name)
        self.optimizer = optimizer

    def compute_gradients(self, loss, var_list=None,
                        gate_gradients=GATE_OP,
                        aggregation_method=None,
                        colocate_gradients_with_ops=False,
                        grad_loss=None):
        assert type(loss) is list
        num_tasks = len(loss)
        loss = tf.stack(loss)
        tf.random.shuffle(loss)

        # Compute per-task gradients.
        grads_task = tf.vectorized_map(lambda x: tf.concat([tf.reshape(grad, [-1,]) 
                            for grad in tf.gradients(x, var_list) 
                            if grad is not None], axis=0), loss)

        # Compute gradient projections.
        def proj_grad(grad_task):
            for k in range(num_tasks):
                inner_product = tf.reduce_sum(grad_task*grads_task[k])
                proj_direction = inner_product / tf.reduce_sum(grads_task[k]*grads_task[k])
                grad_task = grad_task - tf.minimum(proj_direction, 0.) * grads_task[k]
            return grad_task

