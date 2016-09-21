import tensorflow 		as 	tf
import numpy 			as 	np
import datagenerator 	as 	gen


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
		self.sess = tf.Session()
		self.sess.run(tf.initialize_all_variables())
		xEpoch, yEpoch 	= self.system.dataGenerator.generateData(self.system.dataSize)
		numberOfEpochs 	= numberOfEpochs
		dataSize		= self.system.dataSize
		batchSize 		= self.system.batchSize
		testSize		= self.system.testSize

		self.system.printer.printStart()

		for epoch in xrange(numberOfEpochs) :

			self.epochCost = 0
			for i in xrange(dataSize / batchSize) :

				startIndex 	= i*batchSize
				endIndex	= startIndex + batchSize
				xBatch 		= xEpoch[startIndex:endIndex]
				yBatch 		= yEpoch[startIndex:endIndex]
				bOpt, bCost = self.sess.run([self.optimizer, self.cost], 
											 feed_dict={self.x: xBatch, self.y: yBatch})
				self.epochCost += bCost

			if epoch % self.system.testInterval == 0 :
				tOpt, tCost = self.sess.run([self.optimizer, self.cost], 
											 feed_dict={self.x: xEpoch, self.y: yEpoch})
				self.system.printer.printProgress(epoch, tCost)
			else :
				self.system.printer.printProgress(epoch)

