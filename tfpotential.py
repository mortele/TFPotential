import tensorflow 			as 	tf
import numpy 				as 	np
import matplotlib.pyplot 	as 	plt
import datetime 			as 	time
import argumentparser 		as 	ap
import filefinder			as  ff
import neuralnetwork	 	as  nn
import networktrainer		as  nt


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
		

	def __call__(self, inputData) :
		return self.network(inputData)


tfpot = TFPotential()
