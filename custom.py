from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers

def biHyperbolic(x, lmbda, tau_1, tau_2):
    return K.sqrt(1/16*(4 * lmbda * x + 1)**2 + tau_1**2) - K.sqrt(1/16*(4 * lmbda * x - 1)**2 + tau_2**2)
        
class AdaptativeBiHyperbolic(Layer):
    
    def __init__(self, lambda_init='one', tau_init='glorot_normal', mode='asymmetric', **kwargs):
        self.mode = mode
        
        self.lambda_init = initializers.get(lambda_init)     
        self.tau_init = initializers.get(tau_init)

        super(AdaptativeBiHyperbolic, self).__init__(**kwargs)
        
    def build(self, input_shape):    
         
        self.lambdas = self.add_weight(name='lambdas', 
                                      shape=list(input_shape[1:]),
                                      initializer=self.lambda_init,
                                      trainable=True)
        
        self.taus = self.add_weight(name='taus', 
                                      shape=list(input_shape[1:]),
                                      initializer=self.tau_init,
                                      trainable=True)
        
        if self.mode == 'asymmetric':
            self.taus_2 = self.add_weight(name='taus_2', 
                                      shape=list(input_shape[1:]),
                                      initializer=self.tau_init,
                                      trainable=True)
        
        super(AdaptativeBiHyperbolic, self).build(input_shape)   
        
    def call(self, x):
        if self.mode == 'symmetric':
            return biHyperbolic(x, self.lambdas, self.taus, self.taus)
        else:
            return biHyperbolic(x, self.lambdas, self.taus, self.taus_2)
            
    def compute_output_shape(self, input_shape):
        return input_shape

