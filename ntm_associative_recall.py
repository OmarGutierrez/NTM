'''
Modelo: Máquina de Turing Neuronal
Tarea: Llamada asociativa.

Se entrena a la MTN para recibir una secuencia N vectores de N+1 bits y
dos vectores adicionales, el vector N+1 contiene una bandera en el bit N+1
y un 1 en alguno de los N bits restantes (las demás localidades son 0) que 
indica el vector que se desea recuperar. Si el 1 esá en la posición 2,
entonces se desea recuperar el segundo vector. El vector N+1 es únicamente
de ceros.

Basada en el artículo: https://arxiv.org/abs/1410.5401
'''


#Para correr con python 2 y 3
from __future__ import division, print_function, unicode_literals
from __future__ import absolute_import

#Imports
import numpy as np
import os
import tensorflow as tf
import math

import collections
import hashlib
import numbers

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest

#Iniciando sesión interactiva de TensorFlow
sess = tf.InteractiveSession()

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


#Combinación lineal entre capas
#Fuente: https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/python/ops/rnn_cell_impl.py
def _linear(args,
            output_size,
            bias,
            bias_initializer=None,
            kernel_initializer=tf.random_uniform_initializer(-0.5, 0.5)):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_initializer: starting value to initialize the bias
      (default is all zeros).
    kernel_initializer: starting value to initialize the weight.
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape() for a in args]
  for shape in shapes:
    if shape.ndims != 2:
      raise ValueError("linear is expecting 2D arguments: %s" % shapes)
    if shape[1].value is None:
      raise ValueError("linear expects shape[1] to be provided for shape %s, "
                       "but saw %s" % (shape, shape[1]))
    else:
      total_arg_size += shape[1].value

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  scope = vs.get_variable_scope()
  with vs.variable_scope(scope) as outer_scope:
    weights = vs.get_variable(
        _WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size],
        dtype=dtype,
        initializer=kernel_initializer)
    if len(args) == 1:
      res = math_ops.matmul(args[0], weights)
    else:
      res = math_ops.matmul(array_ops.concat(args, 1), weights)
    if not bias:
      return res
    with vs.variable_scope(outer_scope) as inner_scope:
      inner_scope.set_partitioner(None)
      if bias_initializer is None:
        bias_initializer = init_ops.constant_initializer(0.0, dtype=dtype)
      biases = vs.get_variable(
          _BIAS_VARIABLE_NAME, [output_size],
          dtype=dtype,
          initializer=bias_initializer)
    return nn_ops.bias_add(res, biases)


#Convertir la colección de estados en una tupla
#Fuente: https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/python/ops/rnn_cell_impl.py
_StateToTuple = collections.namedtuple("StateToTuple", ("h","M","r","ww","wr"))
class StateToTuple(_StateToTuple):
    """Tuple used by LSTM Cells for `state_size`, `zero_state`, and output state.
    Stores two elements: `(c, h)`, in that order.
    Only used when `state_is_tuple=True`.
    """
    __slots__ = ()


#Clase NTMCell
class NTMCell(tf.nn.rnn_cell.RNNCell):
    '''
    '''

    def __init__(self,
        control_size=100,   #Número de neuronas en el control
        activation=None,    #Función de activación (if None then tanh)
        memory_num_loc=50,  #Número de localidades de memoria
        memory_size_loc=20, #Tamaño de cada localidad de memoria
        reuse=None):
      
        super(NTMCell, self).__init__(_reuse=reuse)
       
        self._num_units = control_size
        self._activation = activation or math_ops.tanh
        self._memory_num_loc = memory_num_loc
        self._memory_size_loc = memory_size_loc

    #Se define el tamaño de cada vector que conformará el estado de la NTM
    @property
    def state_size(self):
        #state: [ [h],[M],[r],[ww],[wr] ]
        #state: [ [num_units],[num_units],[memory_num_loc*memory_size_loc],
        #       [memory_size_loc],[memory_num_loc],[memory_num_loc] ]
        return (StateToTuple(self._num_units,
            self._memory_num_loc*self._memory_size_loc,
            self._memory_size_loc,self._memory_num_loc,self._memory_num_loc))
            
    @property
    def output_size(self):
        return self._num_units


    def call(self, inputs, state):
        """
        Computa un paso de la NTM
        """

        sigmoid = math_ops.sigmoid

        #Dividiendo state
        h,M,r,ww,wr = state

        #Conviertiendo M de vector a matriz
        M = tf.reshape(M,[self._memory_num_loc,self._memory_size_loc])
        


        ################################################################
        ################################################################
                                #Controlador
        ################################################################
        ################################################################
        with vs.variable_scope("controller"):
            new_h = tf.nn.tanh(_linear([inputs, r], self._num_units, True))
            #new_h = tf.Print(new_h, [new_h], message="This is new_h: ")


        ################################################################
        ################################################################
                            #Cabeza de escritura
        ################################################################
        ################################################################
        with vs.variable_scope("write_head"):

            # key vector    k in R^m or [-1,1]^m or [0,inf]^m
            with vs.variable_scope("key"):
                k_w =tf.nn.tanh(_linear([new_h], self._memory_size_loc, True))
                #k_w= tf.Print(k_w, [k_w], message="This is k_w: ")

            # beta factor de instensidad    beta > 0 or > 1
            with vs.variable_scope("beta"):
                #beta_w =tf.add(tf.nn.softplus(_linear([new_h], 1, True)),tf.constant(1.0))
                beta_w =tf.nn.softplus(_linear([new_h], 1, True))
                #beta_w= tf.Print(beta_w, [beta_w], message="This is beta_w: ")

            # interpolation gate    g in [0,1]
            with vs.variable_scope("inter_gate"):
                g_w =tf.sigmoid(_linear([new_h], 1, True))
                #g_w= tf.Print(g_w, [g_w], message="This is g_w: ")

            # shift weight    
            with vs.variable_scope("shift_weight"):
                s_w =tf.nn.softmax(_linear([new_h], 3, True))
                #s_w= tf.Print(s_w, [s_w], message="This is s_w: ")

            # factor de agudizamiento    gamma >= 1
            with vs.variable_scope("fact_agudiz"):
                gamma_w =tf.add( tf.nn.softplus(_linear([new_h], 1, True)), tf.constant(1.0))
                #gamma_w= tf.Print(gamma_w, [gamma_w], message="This is gamma_w: ")

            # erase vector    e in [0,1]^m
            with vs.variable_scope("erase_vector"):
                e_w =tf.nn.sigmoid(_linear([new_h], self._memory_size_loc, True))
                #e_w= tf.Print(e_w, [e_w], message="This is e_w: ")


            # add vector    a in R^m or [-1,1]^m or [0,inf]^m
            with vs.variable_scope("add_vector"):
                a_w =tf.nn.tanh(_linear([new_h], self._memory_size_loc, True))
                #a_w= tf.Print(a_w, [a_w], summarize=self._memory_num_loc, message="This is a_w: ")


        ################################################################
        ################################################################
                            #Cabeza de lectura
        ################################################################
        ################################################################
        with vs.variable_scope("read_head"):

            # key vector    k in R^m or [-1,1]^m or [0,inf]^m
            with vs.variable_scope("key"):
                k_r =tf.nn.tanh(_linear([new_h], self._memory_size_loc, True))
                #k_r= tf.Print(k_r, [k_r], message="This is k_r: ")

            # beta factor de instensidad    beta > 0 or > 1
            with vs.variable_scope("beta"):
                #beta_r =tf.add(tf.nn.softplus(_linear([new_h], 1, True)),tf.constant(1.0))
                beta_r =tf.nn.softplus(_linear([new_h], 1, True))
                #beta_r= tf.Print(beta_r, [beta_r], message="This is beta_r: ")

            # shift weight    
            with vs.variable_scope("shift_weight"):
                s_r =tf.nn.softmax(_linear([new_h], 3, True))
                #s_r= tf.Print(s_r, [s_r], message="This is s_r: ")

            # interpolation gate    g in [0,1]
            with vs.variable_scope("inter_gate"):
                g_r =tf.sigmoid(_linear([new_h], 1, True))
                #g_r= tf.Print(g_r, [g_r], message="This is g_r: ")

            # factor de agudizamiento    gamma >= 1
            with vs.variable_scope("fact_agudiz"):
                gamma_r =tf.add( tf.nn.softplus(_linear([new_h], 1, True)), tf.constant(1.0))
                #gamma_r= tf.Print(gamma_r, [gamma_r], message="This is gamma_r: ")

        ################################################################
        ################################################################
                         #Mecanismo de direccionamiento
        ################################################################
        ################################################################
        def adressing(M,w,k,beta,g,s,gamma):
            ################################################################
            ################################################################
                                #Basado en contenido
            ################################################################
            ################################################################
            #cosine similarity [0]: epsilon para evitar indeterminación
            eps=1e-6
            #cosine similarity [1]: obteniendo u·v
            uv = tf.matmul(k, tf.transpose(M)) #vector fila tamaño N

            #cosine similarity [2]: obteniendo ||v|| (norma de cada M(i))
            norma_M=tf.sqrt(tf.reduce_sum(tf.multiply(M,M), 1)) #vector columna de tamaño N

            #cosine similarity [3]: obteniendo ||k|| 
            norma_k=tf.sqrt(tf.reduce_sum(tf.multiply(k,k), 1)) #escalar

            #cosine similarity [4]: obteniendo ||k|| ||v||
            denominador = tf.transpose(tf.multiply(norma_k,norma_M)+eps) #vector fila de tamaño N

            #cosine similarity [5]: obteniendo u·v / ||k|| ||v||
            S = uv/denominador #vector fila de tamaño N

            #cosine similarity [6]: obteniendo beta*S
            beta_S = beta*S

            #w^c
            wc = tf.nn.softmax(beta_S) #vector fila de tamaño N
            #wc= tf.Print(wc, [wc], message="This is wc: ")



            ################################################################
            ################################################################
                                #Basado en ubicación
            ################################################################
            ################################################################

            ###########################################
            #interpolación
            ###########################################
            #w^g (wc y w_m1 debem ser vectores fila)
            wg = (g*wc) + (1-g)*w #vector fila de tamaño N
            #wg= tf.Print(wg, [wg], message="This is wg: ")


            ###########################################
            #desplazamiento
            ###########################################
            #Fuente: https://github.com/yeoedward/Neural-Turing-Machine/blob/master/test_ntm_rotate.py
            def conv(v, k):
              """Computes circular convolution.
              Args:
                  v: a 1-D `Tensor` (vector)
                  k: a 1-D `Tensor` (kernel)
              """
              size = int(v.get_shape()[0])
              kernel_size = int(k.get_shape()[0])
              kernel_shift = int(math.floor(kernel_size/2.0))

              def loop(idx):
                  if idx < 0: return size + idx
                  if idx >= size : return idx - size
                  else: return idx

              kernels = []
              for i in range(size):
                  indices = [loop(i+j) for j in range(kernel_shift, -kernel_shift-1, -1)]
                  v_ = tf.gather(v, indices)
                  kernels.append(tf.reduce_sum(v_ * k, 0))

              return tf.stack(kernels)


            def desplazamiento(wg, s):
              #nbatches = int(wg.get_shape()[0])
              nbatches = 1
              res = []
              for i in range(nbatches):
                res.append(conv(
                  tf.squeeze(tf.slice(wg, [i, 0], [1, -1])),
                  tf.squeeze(tf.slice(s, [i, 0], [1, -1])),
                ))
              return tf.stack(res)

            w_ = desplazamiento(wg, s) #vector fila de tamaño N
            #w_= tf.Print(w_, [w_], message="This is w_: ")


            ###########################################
            #agudizamiento
            ###########################################
            w_pow = tf.pow(w_,gamma)
            new_w = w_pow / tf.reduce_sum(w_pow, 1, keep_dims=True) #vector fila de tamaño N
            #new_w= tf.Print(new_w, [new_w], message="This is new_w: ")
            return new_w

        


        ################################################################
        ################################################################
                            #operación de escritura
        ################################################################
        ################################################################
        def write(M,a,e,new_w):
            ###########################################
            #erase
            ###########################################
            M_ = tf.multiply(M, (1-(tf.transpose(new_w)*e))) #matrix N X M

            ###########################################
            #add
            ###########################################
            new_M = M_ + tf.transpose(new_w)*a #matrix N X M

            return new_M

        ################################################################
        ################################################################
                                #operación de lectura
        ################################################################
        ################################################################
        def read(new_M,new_w):
            new_r = tf.reduce_sum(tf.multiply(tf.transpose(new_w),new_M),0) #vector fila de tamaño M

            return new_r


        #Obteniendo los nuevos ww y wr
        new_ww=adressing(M,ww,k_w,beta_w,g_w,s_w,gamma_w)
        new_wr=adressing(M,wr,k_r,beta_r,g_r,s_r,gamma_r)

        #Obteniendo nueva actualización de M
        new_M = write(M,a_w,e_w,new_ww)

        #Obteniendo nuevo vector de lectura r
        new_r = read(new_M,new_wr)
        
        #print
        #new_r= tf.Print(new_r, [new_r], summarize=self._memory_num_loc, message="This is new_r: ")
        #new_ww= tf.Print(new_ww, [new_ww], summarize=self._memory_num_loc, message="This is new_ww: ")
        #new_wr= tf.Print(new_wr, [new_wr], summarize=self._memory_num_loc, message="This is new_wr: ")


        #Transformando todos los tensores que pasarán al siguiente estado a forma vectorial
        new_h = new_h
        new_M = tf.reshape(new_M,[1,self._memory_num_loc*self._memory_size_loc])
        new_r = tf.reshape(new_r,[1,self._memory_size_loc])
        new_ww = new_ww
        new_wr = new_wr

        #Juntando los nuevos vectores en una tupla
        new_state = StateToTuple(new_h,new_M,new_r,new_ww,new_wr)

        return new_h, new_state
       


################################################################
################################################################
                #Generación de nuevos datos
################################################################
################################################################
def next_batch(batch_size, length, bits):
    X_sec = np.zeros([batch_size,length,bits])
    X_sec[:,:,0:bits] = np.random.rand(batch_size,length,bits).round()
    X_ntm = np.zeros([batch_size,length+2,bits+1])
    X_ntm[:,length,-1] = 1
    wanted = np.random.random_integers(0, high=length-1, size=batch_size)
    X_ntm[:,length,wanted] = 1

    X_ntm[:,0:length,0:bits] = X_sec[:,:,:]

    Y_ntm = np.zeros([batch_size,length+2,bits])
    Y_ntm[:,-1,:] = X_sec[:,wanted,:]

    return X_ntm, Y_ntm

def next_batch_test(batch_size, length, bits, solicitud):
    X_sec = np.zeros([batch_size,length,bits])
    X_sec[:,:,0:bits] = np.random.rand(batch_size,length,bits).round()
    X_ntm = np.zeros([batch_size,length+2,bits+1])
    X_ntm[:,length,-1] = 1
    #wanted = np.random.random_integers(0, high=length-1, size=batch_size)
    wanted = solicitud
    X_ntm[:,length,wanted] = 1

    X_ntm[:,0:length,0:bits] = X_sec[:,:,:]

    Y_ntm = np.zeros([batch_size,length+2,bits])
    Y_ntm[:,-1,:] = X_sec[:,wanted,:]

    return X_ntm, Y_ntm




length =5
bits = 5
control_size = 100

Epochs = 20000
batch_size = 1

memory_num_loc=50 
memory_size_loc=20

learning_rate = 1e-4
decay=0.9
momentum=0.9
epsilon=1e-10


tar = tf.placeholder(tf.float32, [None, length, bits])
pred = tf.placeholder(tf.float32, [None, length, bits])
def errores(target,prediction):
    num_errores = length*bits - tf.reduce_sum(tf.cast(tf.equal(target,prediction),tf.int32))
    return num_errores

num_errores = errores(tar,pred)




X = tf.placeholder(tf.float32, [None, length+2, bits+1])
y = tf.placeholder(tf.float32, [None, length+2, bits])



cell = tf.contrib.rnn.OutputProjectionWrapper(
    NTMCell(
        control_size=control_size,
        activation=None,
        memory_num_loc = memory_num_loc, 
        memory_size_loc=memory_size_loc),
    
        output_size=bits, activation=None)

weight_init = np.zeros([1, memory_num_loc])


state_init = (
    tf.zeros([1, control_size]),
    tf.ones([1, memory_num_loc*memory_size_loc])*1e-6,
    tf.zeros([1, memory_size_loc]),
    tf.zeros([1, memory_num_loc]),
    tf.zeros([1, memory_num_loc]))
#state_init_batch = tf.tile(state_init, [batch_size, 1])

state_init_batch = StateToTuple(
    tf.zeros([1, control_size]),
    tf.ones([1, memory_num_loc*memory_size_loc])*np.random.normal(loc=0.0, scale=1.0, size=[1, memory_num_loc*memory_size_loc]),
    tf.zeros([1, memory_size_loc]),
    tf.zeros([1, memory_num_loc])+weight_init,
    tf.zeros([1, memory_num_loc])+weight_init)

outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32,initial_state=state_init_batch)




#loss = tf.reduce_mean(tf.square(outputs - y)) # MSE
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=outputs, labels=y), name="loss")
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,decay=decay,momentum=momentum)
training_op = optimizer.minimize(loss)

prediccion_test = tf.round(tf.sigmoid(outputs))

init = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
#saver = tf.train.Saver()


'''
#Restore variables from disk.
saver.restore(sess, "/saved/model.ckpt")
print("Model restored.")
# Do some work with the model
'''


################################################################
################################################################
                #Rutina de entranamiento
################################################################
################################################################
init.run()

for epoch in range(Epochs):
    X_batch, y_batch = next_batch(batch_size, length, bits)
    sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
    if epoch % 10 == 0:
        mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
        print(epoch, "\tMSE:", mse)


# Save the variables to disk.
#save_path = saver.save(sess, "saved/model.ckpt")
#print("Model saved in file: %s" % save_path)



#test
print("\nTest: ")
for i in range(length):
    X_test, Y_test = next_batch_test(1, length, bits, i)
    copia = sess.run(prediccion_test, feed_dict={X: X_test, y: Y_test})

    print("\nX: ")
    print(X_test)
    print("\nVector solicitado (NTM): ")
    print(copia)
    print("\nVector solicitado (Target): ")
    print(Y_test)

