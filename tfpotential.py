import os
import sys
import printer
import plotter
import tensorflow 			as 	tf
import numpy 				as 	np
import matplotlib.pyplot 	as 	plt
import datetime				as 	time
import argumentparser		as 	ap
import filefinder			as  ff
import neuralnetwork		as  nn
import networktrainer		as  nt
import datagenerator 		as  gen
import checkpointsaver		as 	ckps
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#tf.set_random_seed(2)

class TFPotential :
	def __init__(self) :
		self.argumentParser = ap.ArgumentParser(self)
		self.fileFinder 	= ff.FileFinder(self)
		self.inputs		 	= 1
		self.nLayers 	 	= self.argumentParser.nLayers()
		self.nNodes  	 	= self.argumentParser.nNodes()
		self.outputs 	 	= 1
		self.networkType 	= self.argumentParser.type()
		self.network 		= nn.NeuralNetwork(self)
		self.network.constructNetwork(inputs		= self.inputs, 
						 			  nNodes		= self.nNodes,
						 			  nLayers		= self.nLayers, 
						 			  outputs		= self.outputs, 
						 			  networkType	= self.networkType) 
		self.saver 			= ckps.CheckpointSaver(self, 
												   self.argumentParser().save)
		self.networkTrainer = nt.NetworkTrainer(self, self.saver)


		self.function		= lambda r: r/r*np.random.normal(0,1)# +np.sin(7.0*np.pi*r)
		self.function		= lambda r: 1/r**12 - 1/r**6
		self.function		= lambda r: 4*(1.0/(r**12) - 1.0/(r**6)) - 4*(1.0/(2.5**12) - 1.0/(2.5**6))

		
		self.dataGenerator	= gen.DataGenerator(0.87, 2.5, self)
		self.dataGenerator.setFunction(self.function)
		
		if not self.argumentParser().file == None :
			self.dataGenerator.setGeneratorType("file");
		else :
			self.dataGenerator.setGeneratorType("function")
		#self.dataGenerator.setGeneratorType("VMC")
		#self.dataGenerator.setGeneratorType("noise")
		
		
		
		
		self.dataSize  		= int(9987)
		
		
		
		
		
		
		self.numberOfEpochs = int(100)
		self.batchSize		= int(500)
		self.testSize		= self.dataSize #int(600)
		self.testInterval	= 5000
		self.printer		= printer.Printer(self)
		self.printer.printSetup()
		self.plotter 		= plotter.Plotter(self)


	def variableSummaries(self, name, variable) :
		with tf.name_scope('Summaries'):

			mean = tf.reduce_mean(variable)
			tf.summary.scalar('Mean/' + name, mean)

			with tf.name_scope('StandardDeviation'):
				stddev = tf.sqrt(tf.reduce_mean(tf.square(variable - mean)))

			tf.summary.scalar('StandardDeviation/' + name, stddev)
			tf.summary.scalar('Max/' + name, tf.reduce_max(variable))
			tf.summary.scalar('Min/' + name, tf.reduce_min(variable))

			tf.summary.histogram(name, variable)       

	def __call__(self, inputData, expectedOutput=None) :
		if expectedOutput == None :
			expectedOutput = inputData
		return self.sess.run(self.networkTrainer.prediction, 
							 feed_dict={self.networkTrainer.x : inputData,
									    self.networkTrainer.y : expectedOutput})

	def train(self, epochs=-1) :
		numberOfEpochs = self.numberOfEpochs if epochs == -1 else epochs
		self.numberOfEpochs = numberOfEpochs
		self.networkTrainer.trainNetwork(numberOfEpochs)
		self.sess = self.networkTrainer.sess

	def setNetworkType(self, typeString) :
		self.network.parseTypeString(typeString)


if __name__ == "__main__" :
	tfpot = TFPotential()
	#tfpot.inputs = 2
	#tfpot.dataGenerator.a = 0.45
	#tfpot.dataGenerator.b = 1.8
	#tfpot.dataGenerator.generatorType = "SW"
	tfpot.train(tfpot.argumentParser().epochs)
	
	
	
