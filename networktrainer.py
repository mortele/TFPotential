import tensorflow 	as 	tf
import numpy 		as 	np


class NetworkTrainer :
	def __init__(self, system) :
		self.system  	= system
		self.x			= tf.placeholder('float', [None, system.inputs],  name='x')
		self.y			= tf.placeholder('float', [None, system.outputs], name='y')
		self.prediction = system.network(self.x)
		self.cost 		= tf.nn.l2_loss(tf.sub(self.prediction, self.y))
		self.optimizer 	= tf.train.AdamOptimizer().minimize(self.cost)
		self.save 		= system.argumentParser().save
		if self.save :
			self.tf.train.Saver(max_to_keep=None)

	def trainNetwork(self, numberOfEpochs) :
		with tf.Session() as sess :
			sess.run(tf.initialize_all_variables())

			for epoch in range(numberOfEpochs) :
				pass

