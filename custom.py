from keras import backend as K
from keras.engine.topology import Layer
from keras.engine import InputSpec
from keras import initializers

def biHyperbolic(x, lmbda, tau_1, tau_2):
    #return K.sqrt(1/16*(4 * lmbda * x + 1)**2 + tau_1**2) - K.sqrt(1/16*(4 * lmbda * x - 1)**2 + tau_2**2)
    return K.sqrt(lmbda**2 * (x + 1 / (2*lmbda))**2 + tau_1**2) - K.sqrt(lmbda**2 * (x - 1 / (2*lmbda))**2 + tau_2**2)
class AdaptativeBiHyperbolic(Layer):

    def __init__(self, lambda_init='one',
                 tau_init='glorot_normal', 
                 mode='asymmetric',
                 shared_axes=None,
                 **kwargs):

        super(AdaptativeBiHyperbolic, self).__init__(**kwargs)

        self.mode = mode
        self.lambda_init = initializers.get(lambda_init)
        self.tau_init = initializers.get(tau_init)

        if self.mode == 'asymmetric':
            self.tau_2_init = initializers.get(tau_init)

        if shared_axes is None:
            self.shared_axes = None
        elif not isinstance(shared_axes, (list, tuple)):
            self.shared_axes = [shared_axes]
        else:
            self.shared_axes = list(shared_axes)

    def build(self, input_shape):
        
        param_shape = list(input_shape[1:])
        self.param_broadcast = [False] * len(param_shape)
        if self.shared_axes is not None:
            for i in self.shared_axes:
                param_shape[i - 1] = 1
                self.param_broadcast[i - 1] = True

        param_shape = tuple(param_shape)

        self.lambdas = self.add_weight(name='lambdas',
                                      shape=param_shape,
                                      initializer=self.lambda_init,
                                      trainable=True)

        self.taus = self.add_weight(name='taus',
                                      shape=param_shape,
                                      initializer=self.tau_init,
                                      trainable=True)

        if self.mode == 'asymmetric':
            self.taus_2 = self.add_weight(name='taus_2',
                                      shape=param_shape,
                                      initializer=self.tau_2_init,
                                      trainable=True)

        #axes = {}
        #if self.shared_axes:
        #    for i in range(1, len(input_shape)):
        #        if i not in self.shared_axes:
        #            axes[i] = input_shape[i]
        #self.input_spec = InputSpec(ndim=len(input_shape), axes=axes)
        #self.built = True

        super(AdaptativeBiHyperbolic, self).build(input_shape)

    def call(self, x):
        if self.mode == 'symmetric':
            return biHyperbolic(x, self.lambdas, self.taus, self.taus)
        else:
            return biHyperbolic(x, self.lambdas, self.taus, self.taus_2)

