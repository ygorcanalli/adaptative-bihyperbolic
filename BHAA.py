from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
from keras import initializers
from keras.engine import InputSpec

class BHAA(Layer):

	def __init__(self,
		lambda_initializer=initializers.Constant(value=1),
		t1_initializer=initializers.glorot_normal(),
		t2_initializer=initializers.glorot_normal(),
		dominio_0_1=False,
		shared_axes=None, **kwargs):
		super(BHAA, self).__init__(**kwargs)

		self.lambda_initializer = initializers.get(lambda_initializer)
		self.t1_initializer = initializers.get(t1_initializer)
		self.t2_initializer = initializers.get(t2_initializer)
		self.dominio_0_1 = dominio_0_1

		if(shared_axes == None):
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

		# lambda
		self.l = self.add_weight(name='l', shape=param_shape, initializer=self.lambda_initializer)
		# tau 1
		self.t1 = self.add_weight(name='t1', shape=param_shape, initializer=self.t1_initializer)
		# tau 2
		self.t2 = self.add_weight(name='t2', shape=param_shape, initializer=self.t2_initializer)

		axes = {}
		if(self.shared_axes):
			for i in range(1, len(input_shape)):
				if(i not in self.shared_axes):
					axes[i] = input_shape[i]

		self.input_spec = InputSpec(ndim=len(input_shape), axes=axes)
		self.built = True

	def call(self, inputs):
		# bi-hiperbolic assimetric adaptative
		# bhaa(l, t1, t2) = h1 - h2
		# h1 = sqrt(( (l^2)*((x + (1/(2*l)))^2) ) + t1^2)
		# h2 = sqrt(( (l^2)*((x + (1/(2*l)))^2) ) + t2^2)
		denom = 2
		soma = 0
		if(self.dominio_0_1):
			denom = 4
			soma = 0.5

		h1 = K.sqrt( (K.square(self.l)*K.square(inputs + (1.0/(denom*self.l)))) + K.square(self.t1) )
		h2 = K.sqrt( (K.square(self.l)*K.square(inputs - (1.0/(denom*self.l)))) + K.square(self.t2) )
		return (h1 - h2) + soma
