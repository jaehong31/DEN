import tensorflow as tf
import numpy as np
import time
import re, random, collections
from collections import defaultdict
from numpy import linalg as LA
from ops import *

class DEN(object):
    def __init__(self, config):
        self.T = 0
        self.task_indices = []
        self.batch_size = config.batch_size
        self.dims = config.dims
        self.params = dict()
        self.ex_k = config.ex_k
        self.param_trained = set()
        self.n_layers = len(self.dims) - 1
        self.n_classes = config.n_classes
        self.max_iter = config.max_iter
        self.init_lr = config.lr
        self.l1_lambda = config.l1_lambda
        self.l2_lambda = config.l2_lambda
        self.gl_lambda = config.gl_lambda
        self.regular_lambda = config.regular_lambda
        self.early_training = config.max_iter / 10.
        self.time_stamp = dict()

        self.loss_thr = config.loss_thr
        self.spl_thr = config.spl_thr

        for i in range(self.n_layers-1):
            w = self.create_variable('layer%d'%(i+1), 'weight', [self.dims[i], self.dims[i+1]])
            b = self.create_variable('layer%d'%(i+1), 'biases', [self.dims[i+1]])
            self.params[w.name] = w
            self.params[b.name] = b

        self.cur_W, self.prev_W = dict(), dict()
    
    def get_params(self):
        """ Access the parameters """
        mdict = dict()
        for scope_name, param in self.params.items():
            w = self.sess.run(param)
            mdict[scope_name] = w
        return mdict

    def load_params(self, params, top = False, time = 999):
        """ parmas: it contains weight parameters used in network, like ckpt """
        self.params = dict()
        if top:
            # for last layer nodes
            for scope_name, param in params.items():
                scope_name = scope_name.split(':')[0]
                if ('layer%d'%self.n_layers in scope_name) and (('_%d'%self.T) in scope_name):
                    w = tf.get_variable(scope_name, initializer = param, trainable = True)
                    self.params[w.name] = w
                elif 'layer%d'%self.n_layers in scope_name:
                    w = tf.get_variable(scope_name, initializer = param, trainable = False)
                    self.params[w.name] = w
                else:
                    pass
            return ;

        if time == 1:
            self.prev_W = dict()
        for scope_name, param in params.items():
            trainable = True
            if time == 1 and 'layer%d'%self.n_layers not in scope_name:
                self.prev_W[scope_name] = param
            scope_name = scope_name.split(':')[0]
            scope = scope_name.split('/')[0]
            name = scope_name.split('/')[1]
            if (scope == 'layer%d'%self.n_layers) and ('_%d'%self.T) not in name: trainable = False
            if (scope in self.param_trained): trainable = False
            # current task is trainable
            w = tf.get_variable(scope_name, initializer = param, trainable = trainable)
            self.params[w.name] = w
            
    def create_variable(self, scope, name, shape, trainable = True):
        with tf.variable_scope(scope):
            w = tf.get_variable(name, shape, trainable = trainable)
            if 'new' not in w.name:
                self.params[w.name] = w
        return w

    def get_variable(self, scope, name, trainable = True):
        with tf.variable_scope(scope, reuse = True):
            w = tf.get_variable(name, trainable = trainable)
            self.params[w.name] = w
        return w

    def extend_bottom(self, scope, ex_k = 10):
        """ bottom layer expansion. scope is range of layer """
        w = self.get_variable(scope, 'weight')
        b = self.get_variable(scope, 'biases')
        prev_dim = w.get_shape().as_list()[0]
        new_w = self.create_variable('new', 'bw', [prev_dim, ex_k])
        new_b = self.create_variable('new', 'bb', [ex_k])

        expanded_w = tf.concat([w, new_w], 1)
        expanded_b = tf.concat([b, new_b], 0)

        self.params[w.name] = expanded_w
        self.params[b.name] = expanded_b
        return expanded_w, expanded_b

    def extend_top(self, scope, ex_k = 10):
        """ top layer expansion. scope is range of layer """
        if 'layer%d'%self.n_layers == scope:
            # extend for all task layer
            for i in self.task_indices:
                if i == self.T:
                    w = self.get_variable(scope, 'weight_%d'%i, True)
                    b = self.get_variable(scope, 'biases_%d'%i, True)
                    new_w = tf.get_variable('new/n%d'%i, [ex_k, self.n_classes], trainable = True)
                else:
                    w = self.get_variable(scope, 'weight_%d'%i, False)
                    b = self.get_variable(scope, 'biases_%d'%i, False)
                    new_w = tf.get_variable('new/n%d'%i, [ex_k, self.n_classes], 
                                initializer = tf.constant_initializer(0.0), trainable = False)

                expanded_w = tf.concat([w, new_w], 0)
                self.params[w.name] = expanded_w
                self.params[b.name] = b
            return expanded_w, b
        else:
            w = self.get_variable(scope, 'weight')
            b = self.get_variable(scope, 'biases')

            level = int(re.findall(r'layer(\d)', scope)[0])
            expanded_n_units = self.expansion_layer[self.n_layers-level-2] # top-down

            next_dim = w.get_shape().as_list()[1]
            new_w = tf.get_variable(scope + 'new_tw', [self.ex_k, next_dim], trainable = True)

            expanded_w = tf.concat([w, new_w], 0)
            self.params[w.name] = expanded_w
            self.params[b.name] = b
            return expanded_w, b

    def extend_param(self, scope, ex_k):
        if 'layer%d'%self.n_layers == scope:
            for i in self.task_indices:
                if i == self.T: # current task(fragile)
                    w = self.get_variable(scope, 'weight_%d'%i, True)
                    b = self.get_variable(scope, 'biases_%d'%i, True)
                    new_w = tf.get_variable('new_fc/n%d'%i, [ex_k, self.n_classes], trainable = True)
                else:
                    # previous tasks
                    w = self.get_variable(scope, 'weight_%d'%i, False)
                    b = self.get_variable(scope, 'biases_%d'%i, False)
                    new_w = tf.get_variable('new_fc/n%d'%i, [ex_k, self.n_classes], 
                                initializer = tf.constant_initializer(0.0), trainable = False)
                expanded_w = tf.concat([w, new_w], 0)
                self.params[w.name] = expanded_w
                self.params[b.name] = b
            return expanded_w, b
        else:
            w = self.get_variable(scope, 'weight')
            b = self.get_variable(scope, 'biases')

            prev_dim = w.get_shape().as_list()[0]
            next_dim = w.get_shape().as_list()[1]
            # connect bottom to top
            new_w = self.create_variable(scope+'/new_fc', 'bw', [prev_dim, ex_k])
            new_b = self.create_variable(scope+'/new_fc', 'bb', [ex_k])

            expanded_w = tf.concat([w, new_w], 1)
            expanded_b = tf.concat([b, new_b], 0)
            # connect top to bottom
            new_w2 = self.create_variable(scope+'/new_fc', 'tw', [ex_k, next_dim + ex_k])

            expanded_w = tf.concat([expanded_w, new_w2], 0)
            self.params[w.name] = expanded_w
            self.params[b.name] = expanded_b
            return expanded_w, expanded_b

    def build_model(self, task_id, prediction = False, splitting = False, expansion = None):
        bottom = self.X
        if splitting:
            for i in range(1, self.n_layers):
                prev_w = np.copy(self.prev_W_split['layer%d'%i + '/weight:0'])
                cur_w = np.copy(self.cur_W['layer%d'%i + '/weight:0'])
                indices = self.unit_indices['layer%d'%i]
                next_dim = prev_w.shape[1]
                if i >= 2 and i < self.n_layers:
                    below_dim = prev_w.shape[0]
                    below_indices = self.unit_indices['layer%d'%(i-1)]
                    bottom_p_prev_ary, bottom_p_new_ary, bottom_c_prev_ary, bottom_c_new_ary = [], [], [], []
                    for j in range(below_dim):
                        if j in below_indices:
                            bottom_p_prev_ary.append(prev_w[j, :])
                            bottom_p_new_ary.append(cur_w[j, :])
                            bottom_c_prev_ary.append(cur_w[j, :])
                            bottom_c_new_ary.append(cur_w[j, :])
                        else:
                            bottom_p_prev_ary.append(cur_w[j, :])
                            bottom_c_prev_ary.append(cur_w[j, :])
                    prev_w = np.array( bottom_p_prev_ary + bottom_p_new_ary ).astype(np.float32)
                    cur_w = np.array( bottom_c_prev_ary + bottom_c_new_ary ).astype(np.float32)

                prev_ary = []
                new_ary = []
                for j in range(next_dim):
                    if j in indices:
                        prev_ary.append(prev_w[:, j]) 
                        new_ary.append(cur_w[:, j]) # will be expanded
                    else:
                        prev_ary.append(cur_w[:, j])
                # fully connected, L1
                expanded_w = np.array( prev_ary + new_ary ).T.astype(np.float32)
                expanded_b = np.concatenate((self.prev_W_split['layer%d'%i + '/biases:0'], 
                                np.random.rand(len(new_ary)))).astype(np.float32)
                with tf.variable_scope('layer%d'%i):
                    w = tf.get_variable('weight', initializer = expanded_w, trainable = True)
                    b = tf.get_variable('biases', initializer = expanded_b, trainable = True)
                self.params[w.name] = w
                self.params[b.name] = b
                bottom = tf.nn.relu(tf.matmul(bottom, w) + b)
            w, b = self.extend_top('layer%d'%self.n_layers, len(new_ary))
            self.y = tf.matmul(bottom, w) + b
        elif expansion:
            for i in range(1, self.n_layers):
                if i == 1:
                    w, b = self.extend_bottom('layer%d'%i, self.ex_k)
                else:
                    w, b = self.extend_param('layer%d'%i, self.ex_k)
                bottom = tf.nn.relu(tf.matmul(bottom, w) + b)
            w, b = self.extend_param('layer%d'%self.n_layers, self.ex_k)
            self.y = tf.matmul(bottom, w) + b
        elif prediction:
            stamp = self.time_stamp['task%d'%task_id]
            for i in range(1, self.n_layers):
                w = self.get_variable('layer%d'%i, 'weight', False)
                b = self.get_variable('layer%d'%i, 'biases', False)
                w = w[:stamp[i-1], :stamp[i]]
                b = b[:stamp[i]]
                print(' [*] task %d, shape : %s'%(i, w.get_shape().as_list()))
 
                bottom = tf.nn.relu(tf.matmul(bottom, w) + b)

            w = self.get_variable('layer%d'%self.n_layers, 'weight_%d'%task_id, False)
            b = self.get_variable('layer%d'%self.n_layers, 'biases_%d'%task_id, False)
            w = w[:stamp[self.n_layers-1], :stamp[self.n_layers]]
            b = b[:stamp[self.n_layers]]
            self.y = tf.matmul(bottom, w) + b
        else:
            for i in range(1, self.n_layers):
                w = self.get_variable('layer%d'%i, 'weight', True)
                b = self.get_variable('layer%d'%i, 'biases', True)
                bottom = tf.nn.relu(tf.matmul(bottom, w) + b)
            prev_dim = bottom.get_shape().as_list()[1]
            w = self.create_variable('layer%d'%self.n_layers, 'weight_%d'%task_id, [prev_dim, self.n_classes], True)
            b = self.create_variable('layer%d'%self.n_layers, 'biases_%d'%task_id, [self.n_classes], True)
            self.y = tf.matmul(bottom, w) + b

        self.yhat = tf.nn.sigmoid(self.y)
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.y, labels = self.Y))

        if prediction:
            return ;

    def selective_learning(self, task_id, selected_params):
        bottom = self.X
        for i in range(1, self.n_layers):
            with tf.variable_scope('layer%d'%i):
                w = tf.get_variable('weight', initializer = selected_params['layer%d/weight:0'%i])
                b = tf.get_variable('biases', initializer = selected_params['layer%d/biases:0'%i])
            bottom = tf.nn.relu(tf.matmul(bottom, w) + b)
        #last layer
        with tf.variable_scope('layer%d'%self.n_layers):
            w = tf.get_variable('weight_%d'%task_id, 
                    initializer = selected_params['layer%d/weight_%d:0'%(self.n_layers, task_id)])
            b = tf.get_variable('biases_%d'%task_id, 
                    initializer = selected_params['layer%d/biases_%d:0'%(self.n_layers, task_id)])

        self.y = tf.matmul(bottom, w) + b
        self.yhat = tf.nn.sigmoid(self.y)
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.y, labels = self.Y))

    def optimization(self, prev_W, selective = False, splitting = False, expansion = None):
        if selective:
            all_var = [ var for var in tf.trainable_variables() if 'layer%d'%self.n_layers in var.name ]
        else:
            all_var = [ var for var in tf.trainable_variables() ]

        l2_losses = []
        for var in all_var:
            l2_losses.append(tf.nn.l2_loss(var))

        opt = tf.train.AdamOptimizer(self.lr)
        regular_terms = []

        if not splitting and expansion == None:
            for var in all_var:
                if var.name in prev_W.keys():
                    prev_w = prev_W[var.name]
                    regular_terms.append(tf.nn.l2_loss(var-prev_w))
        else:
            for var in all_var:
                if var.name in prev_W.keys():
                    prev_w = prev_W[var.name]
                    if len(prev_w.shape) == 1:
                        sliced = var[:prev_w.shape[0]]
                    else:
                        sliced = var[:prev_w.shape[0], :prev_w.shape[1]]
                    regular_terms.append(tf.nn.l2_loss( sliced - prev_w ))

        losses = self.loss + self.l2_lambda * tf.reduce_sum(l2_losses) + \
                    self.regular_lambda * tf.reduce_sum(regular_terms)

        opt = tf.train.AdamOptimizer(self.lr)
        grads = opt.compute_gradients(losses, all_var)
        apply_grads = opt.apply_gradients(grads, global_step = self.g_step)

        l1_var = [ var for var in tf.trainable_variables() ]
        l1_op_list = []
        with tf.control_dependencies([apply_grads]):
            for var in l1_var:
                th_t = tf.fill(tf.shape(var), tf.convert_to_tensor(self.l1_lambda))
                zero_t = tf.zeros(tf.shape(var))
                var_temp = var - (th_t * tf.sign(var))
                l1_op = var.assign(tf.where(tf.less(tf.abs(var), th_t), zero_t, var_temp))
                l1_op_list.append(l1_op)

        GL_var = [var for var in tf.trainable_variables() if 'new' in var.name and ('bw' in var.name or 'tw' in var.name)]
        gl_op_list = []
        with tf.control_dependencies([apply_grads]):
            for var in GL_var:
                g_sum = tf.sqrt(tf.reduce_sum(tf.square(var), 0))
                th_t = self.gl_lambda
                gw = []
                for i in range(var.get_shape()[1]):
                    temp_gw = var[:, i] - (th_t * var[:, i] / g_sum[i])
                    gw_gl = tf.where(tf.less(g_sum[i], th_t), tf.zeros(tf.shape(var[:, i])), temp_gw)
                    gw.append(gw_gl)
                gl_op = var.assign(tf.stack(gw, 1))
                gl_op_list.append(gl_op)

        with tf.control_dependencies(l1_op_list + gl_op_list):
            self.opt = tf.no_op()

    def set_initial_states(self, decay_step):
        self.g_step = tf.Variable(0., trainable=False)
        self.lr = tf.train.exponential_decay(
                    self.init_lr,           # Base learning rate.
                    self.g_step * self.batch_size,  # Current index into the dataset.
                    decay_step,          # Decay step.
                    0.95,                # Decay rate.
                    staircase=True)
        self.X = tf.placeholder(tf.float32, [None, self.dims[0]])
        self.Y = tf.placeholder(tf.float32, [None, self.n_classes])
        
    def add_task(self, task_id, data):
        trainX, trainY, self.valX, self.valY, testX, testY = data
        self.train_range = np.array(range(len(trainY)))
        data_size = len(trainX)
        self.set_initial_states(data_size)

        expansion_layer = [] # to split
        self.expansion_layer = [0, 0] # new units
        self.build_model(task_id)

        if self.T == 1:
            self.optimization(self.prev_W)
            self.sess.run(tf.global_variables_initializer())
            repeated, c_loss = self.run_epoch(self.opt, self.loss, trainX, trainY, 'Train')
            expansion_layer = [0, 0]
        else:
            """ SELECTIVE LEARN """
            print(' [*] Selective retraining')
            self.optimization(self.prev_W, selective = True)
            self.sess.run(tf.global_variables_initializer())

            repeated, c_loss = self.run_epoch(
                self.opt, self.loss, trainX, trainY, 'Train', selective = True, s_iter = self.early_training)

            params = self.get_params()
            self.destroy_graph()
            self.sess = tf.Session()
            
            # select the units
            selected_prev_params = dict()
            selected_params = dict()
            all_indices = defaultdict(list) # nonzero unis 
            for i in range(self.n_layers, 0, -1):
                if i == self.n_layers:
                    w = params['layer%d/weight_%d:0'%(i, task_id)]
                    b = params['layer%d/biases_%d:0'%(i, task_id)]
                    for j in range(w.shape[0]):
                        if w[j, 0] != 0: 
                            all_indices['layer%d'%i].append(j)
                    selected_params['layer%d/weight_%d:0'%(i, task_id)] = w[np.ix_(all_indices['layer%d'%i], [0])]
                    selected_params['layer%d/biases_%d:0'%(i, task_id)] = b
                else:
                    w = params['layer%d/weight:0'%i]
                    b = params['layer%d/biases:0'%i]
                    top_indices = all_indices['layer%d'%(i+1)]
                    for j in range(w.shape[0]):
                        if np.count_nonzero(w[j, top_indices]) != 0 or i == 1: 
                            all_indices['layer%d'%i].append(j)

                    sub_weight = w[np.ix_(all_indices['layer%d'%i], top_indices)]
                    sub_biases = b[all_indices['layer%d'%(i+1)]]
                    selected_params['layer%d/weight:0'%i] = sub_weight
                    selected_params['layer%d/biases:0'%i] = sub_biases
                    selected_prev_params['layer%d/weight:0'%i] = \
                        self.prev_W['layer%d/weight:0'%i][np.ix_(all_indices['layer%d'%i], top_indices)]
                    selected_prev_params['layer%d/biases:0'%i] = \
                        self.prev_W['layer%d/biases:0'%i][all_indices['layer%d'%(i+1)]]


            # learn only selected params
            self.set_initial_states(data_size)
            self.selective_learning(task_id, selected_params)
            self.optimization(selected_prev_params)
            self.sess.run(tf.global_variables_initializer())
            repeated, c_loss = self.run_epoch(self.opt, self.loss, trainX, trainY, 'Train', print_pred=False)
            _vars = [(var.name, self.sess.run(var)) for var in tf.trainable_variables() if 'layer' in var.name]

            for item in _vars:
                key, values = item
                selected_params[key] = values
    
            # union
            for i in range(self.n_layers, 0, -1):
                if i == self.n_layers:
                    temp_weight = params['layer%d/weight_%d:0'%(i, task_id)]
                    temp_weight[np.ix_(all_indices['layer%d'%i], [0])] = \
                        selected_params['layer%d/weight_%d:0'%(i, task_id)]
                    params['layer%d/weight_%d:0'%(i, task_id)] = temp_weight
                    params['layer%d/biases_%d:0'%(i, task_id)] = selected_params['layer%d/biases_%d:0'%(i, task_id)]
                else:
                    temp_weight = params['layer%d/weight:0'%i]
                    temp_biases = params['layer%d/biases:0'%i]
                    temp_weight[np.ix_(all_indices['layer%d'%i], all_indices['layer%d'%(i+1)])] = \
                        selected_params['layer%d/weight:0'%i]
                    temp_biases[all_indices['layer%d'%(i+1)]] = selected_params['layer%d/biases:0'%i]
                    params['layer%d/weight:0'%i] = temp_weight
                    params['layer%d/biases:0'%i] = temp_biases


            """ Network Expansion """
            if c_loss < self.loss_thr:
                pass
            else:
                # addition
                self.destroy_graph()
                self.sess = tf.Session()
                self.load_params(params)
                self.set_initial_states(data_size)
                self.build_model(task_id, expansion = True)
                self.optimization(self.prev_W, expansion = True)
                self.sess.run(tf.global_variables_initializer())
    
                print(' [*] Network expansion (training)')
                repeated, c_loss = self.run_epoch(self.opt, self.loss, trainX, trainY, 'Train', print_pred=False)
                val_preds = self.sess.run(self.yhat, feed_dict = {self.X: self.valX})
                val_perf = self.get_performance(val_preds, self.valY)

                # delete useless params adding by addition.
                params = self.get_params()
                
                for i in range(self.n_layers-1, 0, -1):
                    prev_layer_weight = params['layer%d/weight:0'%i]
                    prev_layer_biases = params['layer%d/biases:0'%i]
                    useless = []
                    for j in range(prev_layer_weight.shape[1] - self.ex_k, prev_layer_weight.shape[1]):
                        if np.count_nonzero(prev_layer_weight[:, j]) == 0:
                            useless.append(j)
                    cur_layer_weight = np.delete(prev_layer_weight, useless, axis = 1)
                    cur_layer_biases = np.delete(prev_layer_biases, useless)
                    params['layer%d/weight:0'%i] = cur_layer_weight
                    params['layer%d/biases:0'%i] = cur_layer_biases

                    if i == self.n_layers-1:
                        for t in self.task_indices:
                            prev_layer_weight = params['layer%d/weight_%d:0'%(i+1, t)]
                            cur_layer_weight = np.delete(prev_layer_weight, useless, axis = 0)
                            params['layer%d/weight_%d:0'%(i+1, t)] = cur_layer_weight
                    else:
                        prev_layer_weight = params['layer%d/weight:0'%(i+1)]
                        cur_layer_weight = np.delete(prev_layer_weight, useless, axis = 0)
                        params['layer%d/weight:0'%(i+1)] = cur_layer_weight

                    self.expansion_layer[i-1] = self.ex_k - len(useless)

                    print("   [*] Expanding %dth hidden unit, %d unit added, (valid, repeated: %d)" \
                            %(i, self.expansion_layer[i-1], repeated))

                print(' [*] Split & Duplication')
                self.cur_W = params
                # find the highly drifted ones and split
                self.unit_indices = dict()
                for i in range(1, self.n_layers):
                    prev = self.prev_W['layer%d/weight:0'%i]
                    cur = params['layer%d/weight:0'%i]
                    next_dim = prev.shape[1]

                    indices = []
                    cosims = []
                    for j in range(next_dim):
                        cosim = LA.norm(prev[:, j] - cur[:prev.shape[0], j])

                        if cosim > self.spl_thr:
                            indices.append(j)
                            cosims.append(cosim)
                    _temp = np.argsort(cosims)[:self.ex_k]
                    print("   [*] split N in layer%d: %d / %d"%(i, len(_temp), len(cosims)))
                    indices = np.array(indices)[_temp]
                    self.expansion_layer[i-1] += len(indices)
                    expansion_layer.append(len(indices))
                    self.unit_indices['layer%d'%i] = indices

                self.prev_W_split = self.cur_W.copy()
                for key, values in self.prev_W.items():
                    temp = self.prev_W_split[key]
                    if len(values.shape) >= 2:
                        temp[:values.shape[0], :values.shape[1]] = values
                    else:
                        temp[:values.shape[0]] = values
                    self.prev_W_split[key] = temp

                self.destroy_graph()
                self.sess = tf.Session()
                self.load_params(params, top = True)
                self.set_initial_states(data_size)
                self.build_model(task_id, splitting = True)
                self.optimization(self.prev_W, splitting = True)
                self.sess.run(tf.global_variables_initializer())

                repeated, c_loss = self.run_epoch(self.opt, self.loss, trainX, trainY, 'Train')
                val_preds = self.sess.run(self.yhat, feed_dict = {self.X: self.valX})
                val_perf = self.get_performance(val_preds, self.valY)
                print("   [*] split, loss: %.4f, nn_perf: %.4f(valid) repeated: %d"%(c_loss, val_perf, repeated))            

        print("   [*] Total expansions: %s"%self.expansion_layer)

        params = self.get_params()
        # time stamp
        stamp = []
        for i in range(1, self.n_layers+1):
            if i == self.n_layers:
                dim = params['layer%d/weight_%d:0'%(i, task_id)].shape[0]
            else:
                dim = params['layer%d/weight:0'%i].shape[0]
            stamp.append(dim)

        stamp.append(10)
        self.time_stamp['task%d'%task_id] = stamp

        self.destroy_graph()
        self.sess = tf.Session()
        self.load_params(params)

        self.set_initial_states(data_size)
        self.build_model(task_id, prediction = True)
        self.sess.run(tf.global_variables_initializer())
        test_preds, test_loss = self.sess.run([self.yhat, self.loss], 
                                    feed_dict = {self.X: testX, self.Y: testY})
        test_perf = self.get_performance(test_preds, testY)

        self.param_trained.add('layer1')
        self.param_trained.add('layer2')

        print(" [*] Task: %d, nn_test_loss: %.4f, test_perf: %.4f, sparsity(avg): %.4f"
            %(task_id, test_loss, test_perf, self.avg_sparsity(task_id)))
        return test_perf, self.avg_sparsity(task_id), tuple(expansion_layer)


    def run_epoch(self, opt, loss, X, Y, desc = 'Train', selective = False, s_iter = 0, print_pred=True):
        c_iter, old_loss, window_size = s_iter, 999, 10
        loss_window = collections.deque(maxlen = window_size)
        while(self.max_iter > c_iter):

            batch_X, batch_Y = self.data_iteration(X, Y, desc)
            _, c_loss = self.sess.run([opt, loss],
                            feed_dict = {
                                self.X: batch_X,
                                self.Y: batch_Y })
            c_iter += 1
            print_iter = 100

            if desc == 'Train' and c_iter % print_iter == 0:
                val_preds, val_loss = self.sess.run([self.yhat, loss], feed_dict = {
                                            self.X: self.valX,
                                            self.Y: self.valY
                                            })
                loss_window.append(val_loss)
                mean_loss = sum(loss_window) / float(window_size)
                val_perf = self.get_performance(val_preds, self.valY)

                if print_pred == True:
                    print(" [*] iter: %d, val loss: %.4f, val perf: %.4f"%(c_iter, val_loss, val_perf))
                if abs(old_loss-mean_loss) < 1e-6:
                    break
                old_loss = mean_loss

            if selective and c_iter >= self.early_training:
                break

        return c_iter, c_loss

    def data_iteration(self, X, Y, desc = 'Train'):
        if desc == 'Train':
            random.shuffle(self.train_range)
            b_idx = self.train_range[: self.batch_size]
            return X[b_idx], Y[b_idx]
        else:
            return X, Y

    def get_performance(self, p, y):

        perf_list = []
        for _i in range(self.n_classes):
            roc, perf = ROC_AUC(p[:,_i], y[:,_i])
            perf_list.append(perf)
 
        return np.mean(perf_list)

    def predict_perform(self, task_id, X, Y, task_name = None):
        self.X = tf.placeholder(tf.float32, [None, self.dims[0]])
        self.Y = tf.placeholder(tf.float32, [None, self.n_classes])
        self.build_model(task_id, prediction = True)
        self.sess.run(tf.global_variables_initializer())

        test_preds = self.sess.run(self.yhat, feed_dict = {self.X: X})
        test_perf = self.get_performance(test_preds, Y)

        if task_name == None:
            task_name = task_id

        print(" [*] Evaluation, Task:%s, test_acc: %.4f" % (str(task_name), test_perf))
        return test_perf

    def prediction(self, X):
        preds = self.sess.run(self.yhat, feed_dict = {self.X: X})
        return preds

    def destroy_graph(self):
        tf.reset_default_graph()

    def avg_sparsity(self, task_id):
        n_params, zeros = 0, 0
        for idx in range(self.n_layers):
            with tf.variable_scope("layer%d"%(idx+1), reuse = True):
                if idx < (self.n_layers-1):
                    w = tf.get_variable('weight')
                else:
                    w = tf.get_variable('weight_%d'%task_id)
            m_value = self.sess.run(w)
            size = 1.
            shape = m_value.shape
            for dim in shape:
                size = size * dim
            n_params += size
            nzero = float(np.count_nonzero(m_value))
            zeros += (size - nzero)
        return (zeros+1) / (n_params+1)

