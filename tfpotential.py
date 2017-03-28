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

tf.set_random_seed(2)

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
		#self.function		= lambda r: 1/r**6 * (1/r**6 - 1)
		self.function		= lambda r: 0.5*r
		self.dataGenerator	= gen.DataGenerator(0.93, 1.6)
		self.dataGenerator.setFunction(self.function)
		self.dataGenerator.setGeneratorType("function")
		#self.dataGenerator.setGeneratorType("VMC")
		self.numberOfEpochs = int(1000)
		self.dataSize  		= int(1e4)
		self.batchSize		= int(1e3)
		self.testSize		= int(1e4)
		self.testInterval	= 5
		self.printer		= printer.Printer(self)
		self.printer.printSetup()
		self.plotter 		= plotter.Plotter(self)

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
	tfpot.setNetworkType('relu-sigmoid')
	tfpot.train(tfpot.argumentParser().epochs)
