import tensorflow 	as 	tf
import numpy 		as 	np



class NeuralNetwork :
	def __init__(self) :
		pass

	def initializeWeights(self, 
						  shape, 
						  layer, 
						  name) :
		return tf.Variable(tf.zeros(shape), name=name+'%d' % (layer))

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

		if networkType == 'relu-sigmoid' :
			self.hiddenActivation 	= tf.nn.relu
			self.lastActivationa 	= tf.nn.sigmoid
		elif networkType == 'sigmoid' or networkType == None:
			self.hiddenActivation	= tf.nn.sigmoid
			self.lastActivationa	= tf.nn.sigmoid
		elif networkType == 'relu' :
			self.hiddenActivation 	= tf.nn.relu
			self.lastActivationa 	= tf.nn.relu

		self.network = lambda inputData : self.networkSigmoid(
									inputData,
									inputs  		 = self.inputs,
									nLayers 		 = self.nLayers,
									nNodes  		 = self.nNodes,
									outputs 		 = self.outputs,
									hiddenActivation = self.hiddenActivation,
									lastActivationa  = self.lastActivationa) 

	def __call__(self, inputData) :
		return self.network(inputData)

	def network(self,
				inputData,
				inputs,
				nLayers,
				nNodes,
				outputs,
				hiddenActivation,
				lastActivationa) :
		w, b = [], []
		w.append(initializeWeights([inputs, nNodes], 0, 'w'))
		b.append(initializeWeights([inputs], 		 0, 'b'))
		y_ = tf.hiddenActivation(tf.add(tf.matmul(inputData, w[0]), b[0]))

		for layer in range(1, nLayers-1) :
			w.append(initializeWeights([nNodes, nNodes], layer, 'w'))
			b.append(initializeWeights([nNodes], 		 layer, 'b'))
			y_ = tf.hiddenActivation(tf.add(tf.matmul(y_, w[layer]), b[layer]))

		w.append(initializeWeights([nNodes, outputs], nLayers, 'w'))
		b.append(initializeWeights([outputs], 		  nLayers, 'b'))
		y_ = tf.lastActivationa(tf.add(tf.matmul(y_, w[nLayers]), b[nLayers]))		

		w.append(initializeWeights([nNodes, outputs], nLayers+1, 'w'))
		b.append(initializeWeights([outputs], 		  nLayers+1, 'b'))
		return tf.add(tf.matmul(y_, w[nLayers+1]), b[nLayers+1])

















