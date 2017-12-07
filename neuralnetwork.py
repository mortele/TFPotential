import tensorflow 	as 	tf
import numpy 		as 	np


class NeuralNetwork :
	def __init__(self, system) :
		self.system  = system
		self.network = None
		self.hiddenActivation 	= tf.nn.sigmoid
		self.lastActivation 	= tf.nn.sigmoid
		self.summary 			= self.system.variableSummaries

	def initializeBias(self, shape, layer) :
		name = 'b%d' % (layer)
		with tf.name_scope("Biases") :
			bias = tf.Variable(tf.zeros(shape), name=name)
			self.summary(name, bias)
		return bias

	def initializeWeight(self, shape, layer) :
		nIn   = shape[0]
		nOut  = shape[1]
		limit = np.sqrt(6.0 / (nIn + nOut))
		lowerLimit = -limit
		upperLimit =  limit

		name = 'w%d' % (layer)
		with tf.name_scope("Weights") :
			weight = tf.Variable(tf.random_uniform(shape, lowerLimit, upperLimit), 
			   					 name=name); 
			self.summary(name, weight)
		return weight

		#return tf.Variable(tf.random_normal(shape, stddev=0.1),
		#				   name=name+'%d' % (layer))

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
		self.x			 = tf.placeholder('float', [inputs,  None], name='x')
		self.y			 = tf.placeholder('float', [outputs, None], name='y')
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
		#print self.hiddenActivation
		#print self.lastActivation

	def __call__(self, inputData) :
		return self.network(inputData)

	def layer(self, 
			  y_,
			  layerNumber, 
			  activation=None, 
			  inputLayer=False, 
			  outputLayer=False) :

		iSize = self.nNodes if (not inputLayer)  else self.inputs
		jSize = self.nNodes if (not outputLayer) else self.outputs
		self.w.append(self.initializeWeight([iSize, jSize], layerNumber))
		self.b.append(self.initializeBias  ([jSize],        layerNumber))
		y_ = tf.add(tf.matmul(y_, self.w[-1]), self.b[-1])
		return y_ if (activation == None) else activation(y_)


	def fullNetwork(self,
#					inputData,
					y_,
					inputs,
					nLayers,
					nNodes,
					outputs,
					hiddenActivation,
					lastActivation) :

		self.w, self.b = [], []
		self.inputs = inputs
		self.nNodes = nNodes
		self.hiddenActivation = hiddenActivation
		self.lastActivation   = lastActivation

		y_ = self.layer(y_, 0, activation=self.hiddenActivation, inputLayer=True)
		y_ = self.layer(y_, 1, activation=self.hiddenActivation)
		y_ = self.layer(y_, 2, activation=self.hiddenActivation)
		y_ = self.layer(y_, 3, activation=None, 				 outputLayer=True)

		"""
		y_ = self.layer(y_, 0, activation=self.hiddenActivation, inputLayer=True)
		for i in xrange(1, nLayers) :
			y_ = self.layer(y_, i, 	   activation=self.hiddenActivation)
			
		y_ = self.layer(y_, nLayers,   activation=self.lastActivation)
		y_ = self.layer(y_, nLayers+1, activation=None, outputLayer=True)
		"""
		return y_

		"""
		#self.w.append(self.initializeWeights([inputs, nNodes], 0, 'w'))
		#self.b.append(self.initializeWeights([nNodes], 		   0, 'b'))
		#y_ = self.hiddenActivation(tf.add(tf.matmul(inputData, self.w[0]), self.b[0]))

		#for layer in range(1, nLayers) :
		iLimit = inputs
		for layer in range(0, nLayers) :
			self.w.append(self.initializeWeights([iLimit, nNodes], layer, 'w'))
			self.b.append(self.initializeWeights([nNodes], 		   layer, 'b'))
			y_ = self.hiddenActivation(tf.add(tf.matmul(y_, self.w[layer]), self.b[layer]))
			iLimit = nNodes

		self.w.append(self.initializeWeights([nNodes, nNodes], nLayers, 'w'))
		self.b.append(self.initializeWeights([nNodes ], 	   nLayers, 'b'))
		y_ = self.lastActivation(tf.add(tf.matmul(y_, self.w[nLayers]), self.b[nLayers]))
		
		self.w.append(self.initializeWeights([nNodes, outputs], nLayers+1, 'w'))
		self.b.append(self.initializeWeights([outputs], 		nLayers+1, 'b'))
		
		return tf.add(tf.matmul(y_, self.w[nLayers+1]), self.b[nLayers+1])
		"""















