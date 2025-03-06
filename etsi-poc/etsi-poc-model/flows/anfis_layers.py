
import logging
logging.getLogger('tensorflow').disabled = True
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # remove WARNING Messages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.utils import register_keras_serializable
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import MinMaxScaler
import sys


# Clear all previously registered custom objects
get_custom_objects().clear()

class parameters():
    def __init__(self, memberships: list, n_input: int = 3, n_memb: int = 3, batch_size: int = 16, n_epochs: int = 25, memb_func: str = 'gaussian', optimizer: str = 'sgd', loss: str = 'mse', mf_range: tuple = (-2,2)):
        self.n_input = n_input  # no. of Regressors
        self.n_memb = n_memb  # no. of fuzzy memberships
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.memb_func = memb_func  # 'gaussian' / 'gbellmf'
        self.optimizer = optimizer   # sgd / adam /
        self.loss = loss  # mse / mae
        self.mf_range = mf_range
        self.memberships = memberships 
        


# Main Class ANFIS
class ANFIS:
    def __init__(self, memberships: list, n_input: int, n_memb: int, batch_size: int = 16, memb_func: str = 'gaussian', mf_range: tuple = (-2,2), name: str = 'My_Anfis'):
        self.memberships = memberships
        self.n = n_input
        self.m = n_memb
        self.batch_size = batch_size
        self.memb_func = memb_func
        self.range = mf_range
        self.name = name
        self.scaler = None
        self.update_scaler('scaler_training_data.pkl')
        input_ = keras.layers.Input(
            shape=(n_input,), name='inputLayer', batch_size=self.batch_size)
        L1 = FuzzyLayer(mf_range, n_input, n_memb, memb_func, name='fuzzyLayer')(input_)
        L2 = RuleLayer(n_input, n_memb, name='ruleLayer')(L1)
        L3 = NormLayer(name='normLayer')(L2)
        L4 = DefuzzLayer(n_input, n_memb, name='defuzzLayer')(L3, input_)
        L5 = SummationLayer(name='sumLayer')(L4)
        L6 = keras.layers.Activation('tanh', name='outputLayer')(L5)
        self.model = keras.Model(inputs=[input_], outputs=[L6], name=name)
            
        self.update_weights()

    def __call__(self, X):
        if self.is_scaler_set():
            X[:] = self.scaler.transform(X[:])
        #return self.model.predict(X, batch_size=self.batch_size)
        return round(self.model.predict(X, batch_size=self.batch_size))
    
    def predict(self, x):
        return self.__call__(x)        

    
    def update_scaler(self, scaler_filename):
        if os.path.exists(scaler_filename):
            self.scaler = self.load_pickle(scaler_filename)
        else:
            self.scaler = None
    def is_scaler_set(self):
        return self.scaler is not None
    
    def load_pickle(self, filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)
        
    def save(self, model_path, **save_model_kwargs):
        self.model.save(model_path, **save_model_kwargs)



    def update_weights(self):
        # premise parameters (mu&sigma for gaussian // a/b/c for bell-shaped)
        if self.memb_func == 'triangle':
            self.a, self.b, self.c = self.model.get_layer(
                'fuzzyLayer').get_weights()
        if self.memb_func == 'gaussian':
            self.mus, self.sigmas = self.model.get_layer(
                'fuzzyLayer').get_weights()
        elif self.memb_func == 'gbellmf':
            self.a, self.b, self.c = self.model.get_layer(
                'fuzzyLayer').get_weights()
        # consequence parameters
        self.bias, self.weights = self.model.get_layer(
            'defuzzLayer').get_weights()

    def plotmfs(self, show_initial_weights=False):
        n_input = self.n
        n_memb = self.m
        memberships = self.memberships

        if self.memb_func == 'triangle':
            a, b, c = np.around(self.model.get_layer('fuzzyLayer').get_weights(), 2)
            a, b, c = a.reshape((n_memb, n_input, 1)), b.reshape(n_memb, n_input, 1), c.reshape(n_memb, n_input, 1)

            # xn = np.linspace(np.min(c) - 2 * np.max(np.abs(a)),
            #      np.max(c) + 2 * np.max(np.abs(a)), 100).reshape((1, 1, -1))
            xn = np.linspace(-2, 2, 100).reshape((1, 1, -1))
            xn = np.tile(xn, (n_memb, n_input, 1))

            epsilon = 1e-10
            memb_curves = np.maximum(
                np.minimum(
                    (xn - a) / np.maximum(b - a, epsilon),
                    (c - xn) / np.maximum(c - b, epsilon)
                ),
                0
            )


            if show_initial_weights:
                a_init, b_init, c_init = np.around(self.init_weights, 2)
                a_init, b_init, c_init = a_init.reshape((n_memb, n_input, 1)), b_init.reshape(
                    n_memb, n_input, 1), c_init.reshape(n_memb, n_input, 1)
                init_curves = np.maximum(
                np.minimum(
                    (xn - a_init) / np.maximum(b_init - a_init, epsilon),
                    (c_init - xn) / np.maximum(c_init - b_init, epsilon)
                ),
                0
                )
                print(f"initial weights: {self.init_weights}")
                
        if self.memb_func == 'gaussian':
            mus, sigmas = np.around(self.model.get_layer(
                'fuzzyLayer').get_weights(), 2)
            mus, sigmas = mus.reshape(
                (n_memb, n_input, 1)), sigmas.reshape(n_memb, n_input, 1)

            xn = np.linspace(np.min(mus) - 2 * np.max(abs(sigmas)),
                             np.max(mus) + 2 * np.max(abs(sigmas)), 100).reshape((1, 1, -1))
            xn = np.tile(xn, (n_memb, n_input, 1))

            # broadcast all curves in one array
            memb_curves = np.exp(-np.square((xn - mus)) / np.square(sigmas))
            np.save('memb_curves.npy', memb_curves)
            if show_initial_weights:
                mus_init, sigmas_init = np.around(self.init_weights, 2)
                mus_init, sigmas_init = mus_init.reshape(
                    n_memb, n_input, 1), sigmas_init.reshape(n_memb, n_input, 1)
                init_curves = np.exp(-np.square((xn - mus_init)
                                                ) / np.square(sigmas_init))
                print(f"initial weights: {self.init_weights}")
        elif self.memb_func == 'gbellmf':
            a, b, c = np.around(self.model.get_layer(
                'fuzzyLayer').get_weights(), 2)
            a, b, c = a.reshape((n_memb, n_input, 1)), b.reshape(
                n_memb, n_input, 1), c.reshape(n_memb, n_input, 1)

            xn = np.linspace(np.min(c) - 2 * np.max(abs(a)),
                             np.max(c) + 2 * np.max(abs(a)), 100).reshape((1, 1, -1))
            xn = np.tile(xn, (n_memb, n_input, 1))

            # broadcast all curves in one array
            memb_curves = 1 / (1 + np.square((xn - c) / a)**b)

            if show_initial_weights:
                a_init, b_init, c_init = np.around(self.init_weights, 2)
                a_init, b_init, c_init = a_init.reshape((n_memb, n_input, 1)), b_init.reshape(
                    n_memb, n_input, 1), c_init.reshape(n_memb, n_input, 1)
                init_curves = 1 / \
                    (1 + np.square((xn - c_init) / a_init)**b_init)

        elif self.memb_func == 'sigmoid':
            gammas, c = np.around(self.model.get_layer(
                'fuzzyLayer').get_weights(), 2)
            gammas, c = gammas.reshape(
                (n_memb, n_input, 1)), c.reshape(n_memb, n_input, 1)

            xn = np.linspace(np.min(c) - 2 * np.max(abs(c)), np.max(c) + 2 * np.max(
                abs(c)), 100).reshape((1, 1, -1))  # TODO: change confidence bands
            xn = np.tile(xn, (n_memb, n_input, 1))

            # broadcast all curves in one array
            memb_curves = 1 / (1 + np.exp(-gammas * (xn - c)))

            if show_initial_weights:
                gammas_init, c_init = np.around(self.init_weights, 2)
                gammas_init, c_init = gammas_init.reshape(
                    n_memb, n_input, 1), c_init.reshape(n_memb, n_input, 1)
                init_curves = 1 / (1 + np.exp(-gammas_init * (xn - c_init)))

        fig, axs = plt.subplots(nrows=n_input, ncols=1, figsize=(8, self.n * 4))
        fig.subplots_adjust(top=0.9, bottom=0.1, hspace=0.4) 
        fig.suptitle('Membership functions', size=16)
        for n in range(self.n):
            axs[n].grid(True)
            axs[n].set_title(memberships[n])
            for m in range(self.m):
                axs[n].plot(xn[m, n, :], memb_curves[m, n, :])

        if show_initial_weights:  # plot initial membership curve
            for n in range(self.n):
                axs[n].set_prop_cycle(None)  # reset color cycle
                for m in range(self.m):
                    axs[n].plot(xn[m, n, :], init_curves[m, n, :],
                                '--', alpha=.5)
        #plt.show()
        fig.savefig(f"memberships.png")

    def fit(self, X, y, **kwargs):
        # save initial weights in the anfis class
        self.init_weights = self.model.get_layer('fuzzyLayer').get_weights()

        # fit model & update weights in the anfis class
        history = self.model.fit(X, y, **kwargs)
        self.update_weights()

        # clear the graphs
        tf.keras.backend.clear_session()

        return history

    def get_memberships(self, Xs):
        intermediate_layer_model = keras.Model(inputs=self.model.input,
                                               outputs=self.model.get_layer('normLayer').output)

        intermediate_L2_output = intermediate_layer_model.predict(Xs)

        return intermediate_L2_output


# Layer 1
@register_keras_serializable(package="Layers")
class FuzzyLayer(keras.layers.Layer):
    def __init__(self, mf_range, n_input, n_memb, memb_func='gaussian', **kwargs):
        super(FuzzyLayer, self).__init__(**kwargs)
        self.n = n_input
        self.m = n_memb
        self.range = mf_range
        self.memb_func = memb_func
# Custom weight initializer
    def equally_spaced_initializer(self, shape, dtype=tf.float32):
        lower_bound, upper_bound = self.range 
        linspace = tf.reshape(tf.linspace(lower_bound, upper_bound, shape[0]),(-1, 1))
        return tf.Variable(tf.tile(linspace, (1, shape[1])))
    
    def custom_initializer(self):
        lower_bound, upper_bound = self.range
        if self.memb_func == 'triangle':
            n = self.m + 2
            array = np.linspace(lower_bound, upper_bound, n)
            #10% perturbation
            perturbation = 0.1 * array  
            random_perturbation = np.random.uniform(-perturbation, perturbation)
            perturbed_array = array + random_perturbation
            arraya = perturbed_array[:-2]
            arrayc = perturbed_array[2:]
            arrayb = (arraya + arrayc) / 2
            transformed_arraya = tf.Variable(np.repeat(arraya[:, np.newaxis], self.n, axis=1))
            transformed_arrayb = tf.Variable(np.repeat(arrayb[:, np.newaxis], self.n, axis=1))
            transformed_arrayc = tf.Variable(np.repeat(arrayc[:, np.newaxis], self.n, axis=1))
            init_points = []
            init_points.append(transformed_arraya)
            init_points.append(transformed_arrayb)
            init_points.append(transformed_arrayc)

        if self.memb_func == 'gbellmf':
            range_width = upper_bound - lower_bound
            step = range_width / (self.m + 1)
            arrayc = np.linspace(lower_bound + step, upper_bound - step, self.m)            
            #10% perturbation
            perturbation = 0.1 * arrayc  
            random_perturbation = np.random.uniform(-perturbation, perturbation)
            perturbed_arrayc = arrayc + random_perturbation
            arraya = np.full(self.m, step / (2-np.random.rand()))
            arrayb = np.full(self.m, step / (2-np.random.rand()))
            transformed_arrayc = tf.Variable(np.repeat(perturbed_arrayc[:, np.newaxis], self.n, axis=1))
            transformed_arraya = tf.Variable(np.repeat(arraya[:, np.newaxis], self.n, axis=1))
            transformed_arrayb = tf.Variable(np.repeat(arrayb[:, np.newaxis], self.n, axis=1))
            init_points = []
            init_points.append(transformed_arraya)
            init_points.append(transformed_arrayb)
            init_points.append(transformed_arrayc)

        if self.memb_func == 'gaussian':
            range_width = upper_bound - lower_bound
            step = range_width / (self.m + 1)
            mus = np.linspace(lower_bound + step, upper_bound - step, self.m)
            sigmas = np.full(self.m, step / 2)
            transformed_array_mu = tf.Variable(np.repeat(mus[:, np.newaxis], self.n, axis=1))
            transformed_array_sigma = tf.Variable(np.repeat(sigmas[:, np.newaxis], self.n, axis=1)) 
            init_points = []
            init_points.append(transformed_array_mu)
            init_points.append(transformed_array_sigma)

        if self.memb_func == 'sigmoid':
            initializer = self.equally_spaced_initializer(shape=(self.m, self.n)) 
            init_points = []
            init_points.append(initializer)
            init_points.append(initializer)


        return init_points

    def build(self, batch_input_shape):
        self.batch_size = batch_input_shape[0]

        if self.memb_func == 'triangle':
            self.a = self.add_weight(name='a',
                                     shape=(self.m, self.n),
                                     initializer = 'ones',
                                     trainable=True)
            self.b = self.add_weight(name='b',
                                     shape=(self.m, self.n),
                                     initializer = 'ones',
                                     trainable=True)
            self.c = self.add_weight(name='c',
                                     shape=(self.m, self.n),
                                     initializer = 'ones',
                                     trainable=True)
            
        if self.memb_func == 'gbellmf':
            self.a = self.add_weight(name='a',
                                     shape=(self.m, self.n),
                                     initializer = 'ones',
                                     trainable=True)
            self.b = self.add_weight(name='b',
                                     shape=(self.m, self.n),
                                     initializer = 'ones',
                                     trainable=True)
            self.c = self.add_weight(name='c',
                                     shape=(self.m, self.n),
                                     initializer = 'ones',
                                     trainable=True)

        elif self.memb_func == 'gaussian':
            self.mu = self.add_weight(name='mu',
                                    shape=(self.m, self.n),
                                    initializer = 'ones',
                                    trainable=True)
            self.sigma = self.add_weight(name='sigma',
                                    shape=(self.m, self.n),
                                    initializer = 'ones',
                                    trainable=True)

        elif self.memb_func == 'sigmoid':
            self.gamma = self.add_weight(name='gamma',
                                         shape=(self.m, self.n),
                                         initializer = 'ones',
                                         trainable=True)

            self.c = self.add_weight(name='c',
                                     shape=(self.m, self.n),
                                     initializer = 'ones',
                                     trainable=True)

        # Be sure to call this at the end
        super(FuzzyLayer, self).build(batch_input_shape)
        super(FuzzyLayer, self).set_weights(self.custom_initializer())


    def call(self, x_inputs):
        if self.memb_func == 'triangle':

            x_reshaped = tf.reshape(tf.tile(x_inputs, (1, self.m)), (-1, self.m, self.n))
        
        # Ensure b - a and c - b are not zero
            b_minus_a = self.b - self.a
            c_minus_b = self.c - self.b
            left_term = tf.math.divide_no_nan(x_reshaped - self.a, b_minus_a)
            right_term = tf.math.divide_no_nan(self.c - x_reshaped, c_minus_b)
        
        # Handle cases where b - a or c - b is very small
            left_term = tf.where(b_minus_a < 1e-10, tf.zeros_like(left_term), left_term)
            right_term = tf.where(c_minus_b < 1e-10, tf.zeros_like(right_term), right_term)

            L1_output = tf.math.maximum(tf.math.minimum(left_term, right_term), 0)
            #tf.print( self.b, output_stream=sys.stderr)

        if self.memb_func == 'gbellmf':
            L1_output = 1 / (1 +
                             tf.math.pow(
                                 tf.square(tf.subtract(
                                     tf.reshape(
                                         tf.tile(x_inputs, (1, self.m)), (-1, self.m, self.n)), self.c
                                 ) / self.a), self.b)
                             )
        elif self.memb_func == 'gaussian':
            L1_output = tf.exp(-1 *
                               tf.square(tf.subtract(
                                   tf.reshape(
                                       tf.tile(x_inputs, (1, self.m)), (-1, self.m, self.n)), self.mu
                               )) / tf.square(self.sigma))
            
        elif self.memb_func == 'sigmoid':
            L1_output = tf.math.divide(1,
                                       tf.math.exp(-self.gamma *
                                                   tf.subtract(
                                                       tf.reshape(
                                                           tf.tile(x_inputs, (1, self.m)), (-1, self.m, self.n)), self.c)
                                                   )
                                       )
        #tf.print(L1_output, output_stream=sys.stderr)
        return L1_output
    def get_config(self):
        config = super(FuzzyLayer, self).get_config()
        config.update(
            {
                "n_input": self.n,
                "n_memb":self.m,
                "mf_range": self.range,
                "memb_func": self.memb_func,
            }
        )
        return config

# Layer 2
@register_keras_serializable(package="Layers")
class RuleLayer(keras.layers.Layer):
    def __init__(self, n_input, n_memb, **kwargs):
        super(RuleLayer, self).__init__(**kwargs)
        self.n = n_input
        self.m = n_memb
        self.batch_size = None

    def build(self, batch_input_shape):
        self.batch_size = batch_input_shape[0]
        # self.batch_size = tf.shape(batch_input_shape)[0]
        # Be sure to call this at the end
        super(RuleLayer, self).build(batch_input_shape)

    def call(self, input_):
        if not (2 <= self.n <= 6):
            raise ValueError('This ANFIS implementation works with 2 to 6 inputs.')

        reshaped_inputs = [tf.reshape(input_[:, :, i], [self.batch_size] + [1] * i + [-1] + [1] * (self.n - i - 1)) for i in range(self.n)]
    
        L2_output = reshaped_inputs[0]
        for i in range(1, self.n):
            L2_output *= reshaped_inputs[i]
    
        return tf.reshape(L2_output, [self.batch_size, -1])
    
    def get_config(self):
        config = super(RuleLayer, self).get_config()
        config.update(
            {
                "n_input": self.n,
                "n_memb": self.m,
            }
        )
        return config

@register_keras_serializable(package="Layers")
# Layer 3
class NormLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, w):
        w_sum = tf.reshape(tf.reduce_sum(w, axis=1), (-1, 1))
        w_norm = w / w_sum
        return w_norm
    
    def get_config(self):
        config = super(NormLayer, self).get_config()
        return config

@register_keras_serializable(package="Layers")
# Layer 4
class DefuzzLayer(keras.layers.Layer):
    def __init__(self, n_input, n_memb, **kwargs):
        super().__init__(**kwargs)
        self.n = n_input
        self.m = n_memb

        self.CP_bias = self.add_weight(name='Consequence_bias',
                                       shape=(1, self.m ** self.n),
                                       initializer=keras.initializers.RandomUniform(
                                           minval=0, maxval=2),
                                       # initializer = 'ones',
                                       trainable=True)
        self.CP_weight = self.add_weight(name='Consequence_weight',
                                         shape=(self.n, self.m ** self.n),
                                         initializer=keras.initializers.RandomUniform(
                                             minval=0, maxval=2),
                                         # initializer = 'ones',
                                         trainable=True)

    def call(self, w_norm, input_):

        L4_L2_output = tf.multiply(w_norm,
                                   tf.matmul(input_, self.CP_weight) + self.CP_bias)
        return L4_L2_output  # Defuzzyfied Layer

    def get_config(self):
        config = super(DefuzzLayer, self).get_config()
        config.update(
            {
                "n_input": self.n,
                "n_memb": self.m,
            }
        )
        return config

@register_keras_serializable(package="Layers")
# Layer 5
class SummationLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, batch_input_shape):
        self.batch_size = batch_input_shape[0]
        #self.batch_size = tf.shape(batch_input_shape)[0]
        # Be sure to call this at the end
        super(SummationLayer, self).build(batch_input_shape)

    def call(self, input_):
        L5_L2_output = tf.reduce_sum(input_, axis=1)
        L5_L2_output = tf.reshape(L5_L2_output, (-1, 1))
        return L5_L2_output
    
    def get_config(self):
        config = super(SummationLayer, self).get_config()
        return config

    # def compute_L2_output_shape(self, batch_input_shape):
        # return tf.TensorShape([self.batch_size, 1])


#########################################################################################
