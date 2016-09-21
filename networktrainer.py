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
		with tf.Session() as sess :
			sess.run(tf.initialize_all_variables())

			xEpoch, yEpoch 	= system.dataGenerator.generateData(system.dataSize)
			numberOfEpochs 	= system.numberOfEpochs
			dataSize		= system.dataSize
			batchSize 		= system.batchSize

			
			for epoch in range(numberOfEpochs) :
				epochLoss = 0
				
				for i in range(dataSize / batchSize) :
					startIndex 	= i*batchSize
					endIndex	= startIndex + batchSize
					xBatch 		= xEpoch[startIndex:endIndex]
					yBatch 		= yEpoch[startIndex:endIndex]
					bOpt, bCost = sess.run([optimizer, cost], 
										   feed_dict={x: xBatch, y: yBatch})
					epochLoss += bCost
				print "bCost=", bCost

				if epoch % system.testInterval == 0 :
					tOpt, tCost = sess.run([optimizer, cost], 
										   feed_dict={x: xEpoch, y: yEpoch})
					print "tCost=", tCost

