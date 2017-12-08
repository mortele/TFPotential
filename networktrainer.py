import sys
import os
import tensorflow 		as 	tf
import numpy 			as 	np
import datagenerator 	as 	gen
import checkpointsaver	as 	ckps


class NetworkTrainer :
	def __init__(self, system, saver) :
		self.system  	= system
		self.x			= tf.placeholder(	'float', [None, system.inputs], 
											name='x')
		self.y			= tf.placeholder(	'float', [None, system.outputs],
											name='y')
		self.prediction = system.network(self.x)
		self.cost 		= tf.nn.l2_loss(tf.subtract(self.prediction, self.y))
		self.testCost 	= tf.nn.l2_loss(tf.subtract(self.prediction, self.y))		
		self.adam 		= tf.train.AdamOptimizer(learning_rate=0.001)
		self.optimizer 	= self.adam.minimize(self.cost)
		self.save 		= system.argumentParser().save
		self.saver 		= saver

	def trainNetwork(self, numberOfEpochs) :
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())
		loaded = self.saver.loadCheckpoint(	self.system.fileFinder.loadFile, 
											self.sess)
		self.system.printer.printLoad(loaded)
		
		if not loaded :
			xEpoch, yEpoch, xTest, yTest = self.system.dataGenerator.generateData \
															(self.system.dataSize,
															 self.system.testSize)
			self.xTest  = xTest
			self.yTest  = yTest
			self.xTrain = xEpoch
			self.yTrain = yEpoch
		
		xTest = self.xTest
		yTest = self.yTest
		xEpoch = self.xTrain
		yEpoch = self.yTrain
		
		dataSize		= self.system.dataSize
		batchSize 		= self.system.batchSize
		testSize		= self.system.testSize
		
		numberOfEpochs 	= numberOfEpochs
		
		self.system.printer.printStart()

		for epoch in xrange(numberOfEpochs) :
			indices = np.random.choice(dataSize, dataSize, replace=False)
			xEpoch = xEpoch[indices]
			yEpoch = yEpoch[indices]			

			self.epochCost = 0
			for i in xrange(dataSize / batchSize) :

				startIndex 	= i*batchSize
				endIndex	= startIndex + batchSize
				xBatch 		= xEpoch[startIndex:endIndex]
				yBatch 		= yEpoch[startIndex:endIndex]
				bOpt, bCost = self.sess.run([self.optimizer, self.cost], 
											 feed_dict={self.x: xBatch, 
											 			self.y: yBatch})
				self.epochCost += bCost

			tCost = -1
			if epoch % self.system.testInterval == 0 :
				tOpt, tCost = self.sess.run([self.testCost, self.cost], 
											 feed_dict={self.x: xTest, 
														self.y: yTest})
			saved = self.saver.saveCheckpoint(epoch, tCost, self.sess)

			self.system.printer.printProgress(epoch, tCost, saved)

		self.system.saver.saveNetwork(self.sess)
		self.system.plotter.plot()







