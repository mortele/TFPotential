import tensorflow 	as 	tf
import numpy 		as 	np


class NeuralNetwork :
	def __init__(self, system) :
		self.system  = system
		self.network = None
		self.hiddenActivation 	= tf.nn.sigmoid
		self.lastActivation 	= tf.nn.sigmoid

	def initializeWeights(self, 
						  shape, 
						  layer, 
						  name,
						  stddev=np.sqrt(2)) :
		return tf.Variable(tf.random_normal(shape, stddev=stddev),
						   name=name+'%d' % (layer))

	def constructNetwork(self, 
						 inputs, 
						 nNodes, 
						 nLayers, 
						 outputs, 
						 networkType=None) :
		self.networkType = networkType
		self.nLayers 	 = nLayers
		self.nNodes 	 = nNodes
		self.inputs	 	 = inputs
		self.outputs	 = outputs
		self.x			 = tf.placeholder('float', [inputs, None],  name='x')
		self.y			 = tf.placeholder('float', [outputs, None], name='y')
		#self.tmp		 = tf.placeholder('float', [None, None], 	name='tmp')
		self.parseTypeString(networkType)

		self.network = lambda inputData : self.fullNetwork(
									inputData,
									inputs  		 = self.inputs,
									nLayers 		 = self.nLayers,
									nNodes  		 = self.nNodes,
									outputs 		 = self.outputs,
									hiddenActivation = self.hiddenActivation,
									lastActivation   = self.lastActivation) 

	def parseTypeString(self, networkType) :
		self.networkType = networkType
		if networkType == 'relu-sigmoid' :
			self.hiddenActivation 	= tf.nn.relu
			self.lastActivation 	= tf.nn.sigmoid
		elif networkType == 'sigmoid' :
			self.hiddenActivation	= tf.nn.sigmoid
			self.lastActivation		= tf.nn.sigmoid
		elif networkType == 'relu' :
			self.hiddenActivation 	= tf.nn.relu
			self.lastActivation  	= tf.nn.relu
		print self.hiddenActivation
		print self.lastActivation

	def __call__(self, inputData) :
		return self.network(inputData)

	def fullNetwork(self,
					inputData,
					inputs,
					nLayers,
					nNodes,
					outputs,
					hiddenActivation,
					lastActivation) :

		self.w, self.b = [], []
		self.w.append(self.initializeWeights([inputs, nNodes], 0, 'w'))
		self.b.append(self.initializeWeights([nNodes], 		  0, 'b'))
		y_ = self.hiddenActivation(tf.add(tf.matmul(inputData, self.w[0]), self.b[0]))

		for layer in range(1, nLayers) :
			self.w.append(self.initializeWeights([nNodes, nNodes], layer, 'w'))
			self.b.append(self.initializeWeights([nNodes], 		  layer, 'b'))
			y_ = self.hiddenActivation(tf.add(tf.matmul(y_, self.w[layer]), self.b[layer]))

		self.w.append(self.initializeWeights([nNodes, nNodes], nLayers, 'w'))
		self.b.append(self.initializeWeights([nNodes ], 		  nLayers, 'b'))
		y_ = self.lastActivation(tf.add(tf.matmul(y_, self.w[nLayers]), self.b[nLayers]))
		
		self.w.append(self.initializeWeights([nNodes, outputs], nLayers+1, 'w'))
		self.b.append(self.initializeWeights([outputs], 		   nLayers+1, 'b'))
		
		return tf.add(tf.matmul(y_, self.w[nLayers+1]), self.b[nLayers+1])
















