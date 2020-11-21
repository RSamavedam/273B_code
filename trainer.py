# Copyright 2017 Calico LLC
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
"""SeqNN trainer"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
from packaging import version
import pdb

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function


from tensorflow.python.eager import context
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import optimizer

GATE_OP = 1



from basenji import layers
from basenji import metrics 


class Trainer:
  def __init__(self, params, train_data, eval_data, out_dir):
    self.params = params
    self.train_data = train_data
    if type(self.train_data) is not list:
      self.train_data = [self.train_data]
    self.eval_data = eval_data
    if type(self.eval_data) is not list:
      self.eval_data = [self.eval_data]
    self.out_dir = out_dir
    self.compiled = False

    # loss
    self.loss = self.params.get('loss','poisson').lower()
    if self.loss == 'mse':
      self.loss_fn = tf.keras.losses.MSE
    elif self.loss == 'bce':
      self.loss_fn = tf.keras.losses.BinaryCrossentropy()
    else:
      self.loss_fn = tf.keras.losses.Poisson()

    # optimizer
    self.make_optimizer()

    # early stopping
    self.patience = self.params.get('patience', 20)

    # compute batches/epoch
    self.train_epoch_batches = [td.batches_per_epoch() for td in self.train_data]
    self.eval_epoch_batches = [ed.batches_per_epoch() for ed in self.eval_data]
    self.train_epochs_min = self.params.get('train_epochs_min', 1)
    self.train_epochs_max = self.params.get('train_epochs_max', 10000)

    # dataset
    self.num_datasets = len(self.train_data)
    self.dataset_indexes = []
    for di in range(self.num_datasets):
      self.dataset_indexes += [di]*self.train_epoch_batches[di]
    self.dataset_indexes = np.array(self.dataset_indexes)

  def compile(self, seqnn_model):
    for model in seqnn_model.models:
      if self.loss == 'bce':
        model_metrics = [metrics.SeqAUC(curve='ROC'), metrics.SeqAUC(curve='PR')]
      else:
        num_targets = model.output_shape[-1]
        model_metrics = [metrics.PearsonR(num_targets), metrics.R2(num_targets)]
      
      model.compile(loss=self.loss_fn,
                    optimizer=self.optimizer,
                    metrics=model_metrics)
      model.run_eagerly = True
    self.compiled = True

  def fit_keras(self, seqnn_model):
    if not self.compiled:
      self.compile(seqnn_model)

    if self.loss == 'bce':
      early_stop = EarlyStoppingMin(monitor='val_loss', mode='min', verbose=1,
                       patience=self.patience, min_epoch=self.train_epochs_min)
      save_best = tf.keras.callbacks.ModelCheckpoint('%s/model_best.h5'%self.out_dir,
                                                     save_best_only=True, mode='min',
                                                     monitor='val_loss', verbose=1)
      plot_val_164 = TestCallback(self.eval_data[0].dataset)
    else:
      early_stop = EarlyStoppingMin(monitor='val_pearsonr', mode='max', verbose=1,
                       patience=self.patience, min_epoch=self.train_epochs_min)
      save_best = tf.keras.callbacks.ModelCheckpoint('%s/model_best.h5'%self.out_dir,
                                                     save_best_only=True, mode='max',
                                                     monitor='val_pearsonr', verbose=1)
      plot_val_164 = TestCallback(self.eval_data[0].dataset)

    callbacks = [
      early_stop,
      tf.keras.callbacks.TensorBoard(self.out_dir),
      tf.keras.callbacks.ModelCheckpoint('%s/model_check.h5'%self.out_dir),
      save_best,
      plot_val_164]

    seqnn_model.model.fit(
      self.train_data[0].dataset,
      epochs=self.train_epochs_max,
      steps_per_epoch=self.train_epoch_batches[0],
      callbacks=callbacks,
      validation_data=self.eval_data[0].dataset,
      validation_steps=self.eval_epoch_batches[0])

  def fit2(self, seqnn_model):
    if not self.compiled:
      self.compile(seqnn_model)

    assert(len(seqnn_model.models) >= self.num_datasets)

    ################################################################
    # prep

    # metrics
    train_loss, train_r, train_r2 = [], [], []
    for di in range(self.num_datasets):
      num_targets = seqnn_model.models[di].output_shape[-1]
      train_loss.append(tf.keras.metrics.Mean())
      train_r.append(metrics.PearsonR(num_targets))
      train_r2.append(metrics.R2(num_targets))

    # generate decorated train steps
    """
    train_steps = []
    for di in range(self.num_datasets):
      model = seqnn_model.models[di]

      @tf.function
      def train_step(x, y):
        with tf.GradientTape() as tape:
          pred = model(x, training=tf.constant(True))
          loss = self.loss_fn(y, pred) + sum(model.losses)
        train_loss[di](loss)
        train_r[di](y, pred)
        train_r2[di](y, pred)
        gradients = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

      train_steps.append(train_step)
    """
    @tf.function
    def train_step0(x, y):
      with tf.GradientTape() as tape:
        pred = seqnn_model.models[0](x, training=tf.constant(True))
        loss = self.loss_fn(y, pred) + sum(seqnn_model.models[0].losses)
      train_loss[0](loss)
      train_r[0](y, pred)
      train_r2[0](y, pred)
      gradients = tape.gradient(loss, seqnn_model.models[0].trainable_variables)
      self.optimizer.apply_gradients(zip(gradients, seqnn_model.models[0].trainable_variables))

    if self.num_datasets > 1:
      @tf.function
      def train_step1(x, y):
        with tf.GradientTape() as tape:
          pred = seqnn_model.models[1](x, training=tf.constant(True))
          loss = self.loss_fn(y, pred) + sum(seqnn_model.models[1].losses)
        train_loss[1](loss)
        train_r[1](y, pred)
        train_r2[1](y, pred)
        gradients = tape.gradient(loss, seqnn_model.models[1].trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, seqnn_model.models[1].trainable_variables))

    # improvement variables
    valid_best = [np.inf]*self.num_datasets
    unimproved = [0]*self.num_datasets

    ################################################################
    # training loop

    for ei in range(self.train_epochs_max):
      if ei >= self.train_epochs_min and np.min(unimproved) > self.patience:
        break
      else:
        # shuffle datasets
        np.random.shuffle(self.dataset_indexes)

        # get iterators
        train_data_iters = [iter(td.dataset) for td in self.train_data]

        # train
        t0 = time.time()
        for di in self.dataset_indexes:
          x, y = next(train_data_iters[di])
          # train_steps[di](x, y)
          if di == 0:
            train_step0(x, y)
          else:
            train_step1(x, y)

        print('Epoch %d - %ds' % (ei, (time.time()-t0)))
        for di in range(self.num_datasets):
          print('  Data %d' % di, end='')
          model = seqnn_model.models[di]

          # print training accuracy
          print(' - train_loss: %.4f' % train_loss[di].result().numpy(), end='')
          print(' - train_r: %.4f' %  train_r[di].result().numpy(), end='')
          print(' - train_r: %.4f' %  train_r2[di].result().numpy(), end='')

          # print validation accuracy
          valid_stats = model.evaluate(self.eval_data[di].dataset, verbose=0)
          print(' - valid_loss: %.4f' % valid_stats[0], end='')
          print(' - valid_r: %.4f' % valid_stats[1], end='')
          print(' - valid_r2: %.4f' % valid_stats[2], end='')
          early_stop_stat = valid_stats[1]

          # checkpoint
          model.save('%s/model%d_check.h5' % (self.out_dir, di))

          # check best
          if early_stop_stat > valid_best[di]:
            print(' - best!', end='')
            unimproved[di] = 0
            valid_best[di] = early_stop_stat
            model.save('%s/model%d_best.h5' % (self.out_dir, di))
          else:
            unimproved[di] += 1
          print('', flush=True)

          # reset metrics
          train_loss[di].reset_states()
          train_r[di].reset_states()
          train_r2[di].reset_states()

        
  def fit_tape(self, seqnn_model):
    if not self.compiled:
      self.compile(seqnn_model)
    model = seqnn_model.model
    print(len(model.trainable_variables))
    self.test_data = list(self.eval_data[0].dataset.as_numpy_iterator())[0]
    num_infractions = np.zeros(164)
    # metrics
    num_targets = model.output_shape[-1]
    train_loss = tf.keras.metrics.Mean()
    train_r = metrics.PearsonR(num_targets)
    train_r2 = metrics.R2(num_targets)
    self.better_optimizer = PCGrad(self.optimizer)
    jacobian_variable = tf.constant([1, 2, 3, 4, 5])
    shutdownCount = 0 

    #@tf.function
    #def calculate_multitask_gradients(task_loss):
        #return tf.vectorized_map(lambda x: tf.gradients(x, model.trainable_variables), task_loss)

    def better_train_step(x, y):
        pred = model(x, training=tf.constant(True))
        loss = self.loss_fn(y, pred) + sum(model.losses)
        flattened_pred = tf.unstack(pred, num=164, axis=2)
        task_pred = []
        for elt in flattened_pred:
            task_pred.append(elt)
        flattened_actual = tf.unstack(y, num=164, axis=2)
        task_actual = []
        for elt in flattened_actual:
            task_actual.append(elt)
        task_loss = []
        for i in range(164):
            task_loss.append(self.loss_fn(task_actual[i], task_pred[i]) + (sum(model.losses) / 164))
        task_loss = tf.stack(task_loss)
        #grads_task = tf.vectorized_map(lambda x: tf.gradients(x, model.trainable_variables), task_loss)
        #grads_task = calculate_multitask_gradients(task_loss)
        #self.optimizer.apply_gradients(zip(grads_task[3], model.trainable_variables))
    
    #@tf.function
    #def train_step(x, y):
      #with tf.GradientTape(persistent=True) as tape:
        #pred = model(x, training=tf.constant(True))
        #loss = self.loss_fn(y, pred) + sum(model.losses)
        #flattened_pred = tf.unstack(pred, num=164, axis=2)
        #task_pred = []
        #for elt in flattened_pred:
            #task_pred.append(elt)
        #flattened_actual = tf.unstack(y, num=164, axis=2)
        #task_actual = []
        #for elt in flattened_actual:
            #task_actual.append(elt)
        #task_loss = []
        #for i in range(164):
            #task_loss.append(self.loss_fn(task_actual[i], task_pred[i]) + (sum(model.losses) / 164))
        #alternate_task_loss = tf.vectorized_map(lambda x: tf.square(x), y - pred)
        #alternate_task_loss = (y - pred)*(y - pred)
        #print(task_loss)
        #task_loss = tf.stack(task_loss)
        #grads_task = tf.vectorized_map(lambda x: tape.gradient(x, model.trainable_variables), task_loss)
        #quantity = task_loss[0] # + task_loss[1]
        #task_loss = tf.stack(task_loss)
      #gradients = tape.gradient(quantity, model.trainable_variables) #this works for some reason
      #print(tf.shape(gradients[0]))
      #self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
      #print(alternate_task_loss)
      #jacobian = tape.jacobian(task_loss, model.trainable_variables, experimental_use_pfor=True)
      #return jacobian
      #i = tf.constant(1, dtype=tf.int32)
      #while tf.less(i, 29):

      #task_0_grads = []
      #for grad in jacobian:
          #task_0_grads.append(tf.unstack(grad)[0])
      #grads_task = tf.unstack(more_gradients, num=164, axis=1)
      #print(tf.shape(more_gradients[0]))
      #task_loss = tf.stack(task_loss)
      #tape.watch(task_loss)
      #grads_task = tf.vectorized_map(lambda x: tape.gradient(x, model.trainable_variables), alternate_task_loss)
      #print(grads_task)
      #self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
      #return more_gradients
      #grads_and_vars = self.better_optimizer.compute_gradients(task_loss, model.trainable_variables)
      #self.better_optimizer.apply_gradients(grads_and_vars)

    # improvement variables
    valid_best = -np.inf
    unimproved = 0
    vectorSet = []
    myArr = np.ones(164, dtype=np.float32)
    #myArr[0] = 0
    print(myArr)
    
    @tf.function
    def train_step_0(x, y, weights):
        with tf.GradientTape(persistent=True) as tape:
            pred = model(x, training=tf.constant(True))
            flattened_pred = tf.unstack(pred, num=164, axis=2)
            task_pred = []
            for elt in flattened_pred:
                task_pred.append(elt)
            flattened_actual = tf.unstack(y, num=164, axis=2)
            task_actual = []
            for elt in flattened_actual:
                task_actual.append(elt)
            task_loss = []
            unweighted_task_loss = []
            for i in range(164):
                task_loss.append((self.loss_fn(task_actual[i], task_pred[i]) + (sum(model.losses) / 164))*(1 / (2 * weights[i] * weights[i])))
                unweighted_task_loss.append(self.loss_fn(task_actual[i], task_pred[i]) + (sum(model.losses) / 164))
            task_loss = tf.stack(task_loss)
            #print(task_loss)
            #print(weights)
            #weighted_sum = tf.tensordot(task_loss, weights)
            weighted_sum = tf.math.reduce_sum(task_loss)
            #weight_gradients = []
            #for i in range(164):
                #weight_gradients.append(unweighted_task_loss[i] * ( -1 / (weights[i] * weights[i] * weights[i])) + tf.math.log(weights[i]))
        gradients = tape.gradient(weighted_sum, model.trainable_variables)
        #print(gradients)
        #print(None in gradients)
        self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return unweighted_task_loss



    # training loop
    for ei in range(self.train_epochs_max):
      if ei >= self.train_epochs_min and unimproved > self.patience:
        break
      else:
        # train
        print("epoch started")
        t0 = time.time()
        train_iter = iter(self.train_data[0].dataset)
        print("num iterations")
        print(self.train_epoch_batches[0])
        for si in range(self.train_epoch_batches[0]):
          x, y = next(train_iter)
          #x = x.numpy()
          #y = y.numpy()
          #print(type(x))
          #print(type(y))
          #myArr = np.ones(164, dtype=np.float32)
          #weights = tf.convert_to_tensor(myArr)
          gradient_weights = train_step_0(x, y, myArr)
          #model.train_on_batch(x, y)
          #print("train step completed")
          #print(si)
          if si % 250 == 0:
              print("250 train steps completed")
              print("weights")
              print(myArr)
              test_x = self.test_data[0]
              test_y = self.test_data[1] 
              predictions = model.predict(test_x)
              squared_differences = tf.square(predictions - test_y)
              sumVector = squared_differences[0]
              for i in range(1, len(squared_differences)):
                  sumVector = sumVector + squared_differences[i]
              sumVector = sumVector / len(squared_differences)
              print("End of epoch validation MSE for each of the 164 cell types: ")
              vectorString = ""
              for entry in sumVector:
                  vectorString = vectorString + str(entry) + " "
              print(vectorString)
              #print(sumVector)
              print(si // 250)
              outputFile = open("164_cell_validation_error.txt", "a")
              outputFile.write(vectorString + "\n")
              outputFile.close()
              outputFile = open("164_cell_weights.txt", "a")
              outputFile.write(str(myArr) + "\n")
              outputFile.close()
              vectorSet.append(sumVector)
              parameter_value = 200
              if len(vectorSet) > parameter_value:
                  vectorIterationIndex = len(vectorSet) - 1
                  checkIndex = vectorIterationIndex - parameter_value
                  for i in range(164):
                      num1 = vectorSet[vectorIterationIndex].numpy()[0][i]
                      #num1 = int(vectorSet[vectorIterationIndex][i])
                      num2 = vectorSet[checkIndex].numpy()[0][i]
                      if num1 > num2:
                          num_infractions[i] += 1
                  for i in range(164):
                      num = int(num_infractions[i])
                      if num > 5:
                          if myArr[i] != 0:
                              shutdownCount += 1
                              if shutdownCount == 82:
                                  outputFile = open("task_being_analyzed.txt", "a")
                                  outputFile.write(str(i) + "\n")
                                  myArr = np.zeros(164, dtype=np.float32)
                                  myArr[i] = 1
                              if shutdownCount < 82:
                                  myArr[i] = 0
                              outputFile = open("early_stopping_stats.txt", "a")
                              outputFile.write(str(i) + ": " + str(len(vectorSet)) + "\n")
                              outputFile.close()





        # print training accuracy
        outputFile = open("epoch_stats.txt", "a")
        print("training for epoch completed")
        train_loss_epoch = train_loss.result().numpy()
        train_r_epoch = train_r.result().numpy()
        print('Epoch %d - %ds - train_loss: %.4f - train_r: %.4f' % (ei, (time.time()-t0), train_loss_epoch, train_r_epoch), end='')
        outputFile.write("training for epoch completed\n")
        outputString = 'Epoch %d - %ds - train_loss: %.4f - train_r: %.4f' % (ei, (time.time()-t0), train_loss_epoch, train_r_epoch)
        outputFile.write(outputString + "\n")

        # checkpoint
        seqnn_model.save('%s/model_check.h5'%self.out_dir)

        # print validation accuracy
        valid_loss, valid_pr, valid_r2 = model.evaluate(self.eval_data[0].dataset, verbose=0)
        print(' - valid_loss: %.4f - valid_r: %.4f - valid_r2: %.4f' % (valid_loss, valid_pr, valid_r2), end='')
        outputString = ' - valid_loss: %.4f - valid_r: %.4f - valid_r2: %.4f' % (valid_loss, valid_pr, valid_r2) 
        outputFile.write(outputString + "\n")
        outputFile.close()

        # check best
        if valid_pr > valid_best:
          print(' - best!', end='')
          unimproved = 0
          valid_best = valid_pr
          seqnn_model.save('%s/model_best.h5'%self.out_dir)
        else:
          unimproved += 1
        print('', flush=True)

        # reset metrics
        train_loss.reset_states()
        train_r.reset_states()

  def make_optimizer(self):
    # schedule (currently OFF)
    initial_learning_rate = self.params.get('learning_rate', 0.01)
    if False:
      lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=self.params.get('decay_steps', 100000),
        decay_rate=self.params.get('decay_rate', 0.96),
        staircase=True)
    else:
      lr_schedule = initial_learning_rate

    if version.parse(tf.__version__) < version.parse('2.2'):
      clip_norm_default = 1000000
    else:
      clip_norm_default = None
    clip_norm = self.params.get('clip_norm', clip_norm_default)

    # optimizer
    optimizer_type = self.params.get('optimizer', 'sgd').lower()
    if optimizer_type == 'adam':
      self.optimizer = tf.keras.optimizers.Adam(
          lr=lr_schedule,
          beta_1=self.params.get('adam_beta1',0.9),
          beta_2=self.params.get('adam_beta2',0.999),
          clipnorm=clip_norm)

    elif optimizer_type in ['sgd', 'momentum']:
      self.optimizer = tf.keras.optimizers.SGD(
          lr=lr_schedule,
          momentum=self.params.get('momentum', 0.99),
          clipnorm=clip_norm)

    else:
      print('Cannot recognize optimization algorithm %s' % optimizer_type)
      exit(1)

class EarlyStoppingMin(tf.keras.callbacks.EarlyStopping):
  """Stop training when a monitored quantity has stopped improving.
  Arguments:
      min_epoch: Minimum number of epochs before considering stopping.
      
  """
  def __init__(self, min_epoch=0, **kwargs):
    super(EarlyStoppingMin, self).__init__(**kwargs)
    self.min_epoch = min_epoch

  def on_epoch_end(self, epoch, logs=None):
    current = self.get_monitor_value(logs)
    if current is None:
      return
    if self.monitor_op(current - self.min_delta, self.best):
      self.best = current
      self.wait = 0
      if self.restore_best_weights:
        self.best_weights = self.model.get_weights()
    else:
      self.wait += 1
      if epoch >= self.min_epoch and self.wait >= self.patience:
        self.stopped_epoch = epoch
        self.model.stop_training = True
        if self.restore_best_weights:
          if self.verbose > 0:
            print('Restoring model weights from the end of the best epoch.')
          self.model.set_weights(self.best_weights)

class TestCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_data):
        self.test_data = list(test_data.as_numpy_iterator())[0]
        self.numBatches = 0
    def on_train_batch_end(self, epoch, logs=None):
        self.numBatches += 1
        self.numBatches = self.numBatches % 250
        if self.numBatches == 0:
            x = self.test_data[0]
            y = self.test_data[1]
            predictions = self.model.predict(x)
            squared_differences = tf.square(predictions - y)
            sumVector = squared_differences[0]
            for i in range(1, len(squared_differences)):
                sumVector = sumVector + squared_differences[i]
            sumVector = sumVector / len(squared_differences)
            print("End of epoch validation MSE for each of the 164 cell types: ")
            vectorString = ""
            for entry in sumVector:
                vectorString = vectorString + str(entry) + " "
            print(vectorString) 
            outputFile = open("164_cell_validation_error", "a")
            outputFile.write(vectorString + "\n")
            outputFile.close()


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

        proj_grads_flatten = tf.vectorized_map(proj_grad, grads_task)

        # Unpack flattened projected gradients back to their original shapes.
        proj_grads = []
        for j in range(num_tasks):
            start_idx = 0
            for idx, var in enumerate(var_list):
                grad_shape = var.get_shape()
                flatten_dim = np.prod([grad_shape.dims[i].value for i in range(len(grad_shape.dims))])
                proj_grad = proj_grads_flatten[j][start_idx:start_idx+flatten_dim]
                proj_grad = tf.reshape(proj_grad, grad_shape)
                if len(proj_grads) < len(var_list):
                    proj_grads.append(proj_grad)
                else:
                    proj_grads[idx] += proj_grad               
                start_idx += flatten_dim
        grads_and_vars = list(zip(proj_grads, var_list))
        return grads_and_vars

    def _create_slots(self, var_list):
        self.optimizer._create_slots(var_list)

    def _prepare(self):
        self.optimizer._prepare()

    def _apply_dense(self, grad, var):
        return self.optimizer._apply_dense(grad, var)

    def _resource_apply_dense(self, grad, var):
        return self.optimizer._resource_apply_dense(grad, var)

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        return self.optimizer._apply_sparse_shared(grad, var, indices, scatter_add)

    def _apply_sparse(self, grad, var):
        return self.optimizer._apply_sparse(grad, var)

    def _resource_scatter_add(self, x, i, v):
        return self.optimizer._resource_scatter_add(x, i, v)

    def _resource_apply_sparse(self, grad, var, indices):
        return self.optimizer._resource_apply_sparse(grad, var, indices)

    def _finish(self, update_ops, name_scope):
        return self.optimizer._finish(update_ops, name_scope)

    def _call_if_callable(self, param):
        """Call the function if param is callable."""
        return param() if callable(param) else param
