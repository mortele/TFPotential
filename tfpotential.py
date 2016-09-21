import sys
import printer
import tensorflow 			as 	tf
import numpy 				as 	np
import matplotlib.pyplot 	as 	plt
import datetime				as 	time
import argumentparser		as 	ap
import filefinder			as  ff
import neuralnetwork		as  nn
import networktrainer		as  nt
import datagenerator 		as  gen



class TFPotential :
	def __init__(self) :
		self.argumentParser = ap.ArgumentParser(self)
		self.filefinder 	= ff.FileFinder(self)
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
						 			  networkType	= None) 
		self.networkTrainer = nt.NetworkTrainer(self)
		self.function		= lambda r: 1/r**6 * (1/r**6 - 1)
		self.dataGenerator	= gen.DataGenerator(0.87, 1.6)
		self.dataGenerator.setFunction(self.function)
		self.numberOfEpochs = int(1e10)
		self.dataSize  		= int(1e7)
		self.batchSize		= int(1e5)
		self.testSize		= int(1e7)
		self.testInterval	= 10
		self.printer		= printer.Printer(self)
		self.printer.printSetup()

	def __call__(self, inputData, expectedOutput=None) :
		if expectedOutput == None :
			expectedOutput = inputData
		return self.sess.run(self.networkTrainer.prediction, 
							 feed_dict={self.networkTrainer.x : inputData,
									    self.networkTrainer.y : expectedOutput})

	def train(self, epochs=-1) :
		numberOfEpochs = self.numberOfEpochs if epochs == -1 else epochs
		self.networkTrainer.trainNetwork(numberOfEpochs)
		self.sess = self.networkTrainer.sess

	def setNetworkType(self, typeString) :
		self.network.parseTypeString(typeString)


if __name__ == "__main__" :
	tfpot = TFPotential()
	tfpot.setNetworkType('relu-sigmoid')
	tfpot.train()
	x,y=tfpot.dataGenerator.generateData(10)
	print tfpot(x)
