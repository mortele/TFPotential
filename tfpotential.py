import tensorflow 			as 	tf
import numpy 				as 	np
import matplotlib.pyplot 	as 	plt
import datetime 			as 	time
import argumentparser 		as 	ap
import filefinder			as  ff
import neuralnetwork	 	as  nn
import networktrainer		as  nt
import dataGenerator 		as  gen


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
		self.numberOfEpochs = sys.maxint
		self.dataSize  		= int(1e7)
		self.batchSize		= int(1e5)
		self.testSize		= int(1e7)
		self.testInterval	= 10

	def __call__(self, inputData) :
		return self.network(inputData)


tfpot = TFPotential()
