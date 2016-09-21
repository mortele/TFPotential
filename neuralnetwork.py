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
						  name) :
		return tf.Variable(tf.random_normal(shape), stddev=np.sqrt(2),
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
		self.x			 = tf.placeholder('float', [None, inputs],  name='x')
		self.y			 = tf.placeholder('float', [None, outputs], name='y')
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
		if networkType == 'relu-sigmoid' :
			self.hiddenActivation 	= tf.nn.relu
			self.lastActivation 	= tf.nn.sigmoid
		elif networkType == 'sigmoid' :
			self.hiddenActivation	= tf.nn.sigmoid
			self.lastActivation		= tf.nn.sigmoid
		elif networkType == 'relu' :
			self.hiddenActivation 	= tf.nn.relu
			self.lastActivation  	= tf.nn.relu

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
		w, b = [], []
		w.append(self.initializeWeights([inputs, nNodes], 0, 'w'))
		b.append(self.initializeWeights([inputs], 		  0, 'b'))
		y_ = self.hiddenActivation(tf.add(tf.matmul(inputData, w[0]), b[0]))

		for layer in range(1, nLayers-1) :
			w.append(self.initializeWeights([nNodes, nNodes], layer, 'w'))
			b.append(self.initializeWeights([nNodes], 		  layer, 'b'))
			y_ = self.hiddenActivation(tf.add(tf.matmul(y_, w[layer]), b[layer]))

		w.append(self.initializeWeights([nNodes, nNodes], nLayers-1, 'w'))
		b.append(self.initializeWeights([nNodes], 		  nLayers-1, 'b'))
		y_ = self.lastActivation(tf.add(tf.matmul(y_, w[nLayers-1]), b[nLayers-1]))

		w.append(self.initializeWeights([nNodes, outputs], nLayers, 'w'))
		b.append(self.initializeWeights([outputs], 		   nLayers, 'b'))
		return tf.add(tf.matmul(y_, w[nLayers]), b[nLayers])

















