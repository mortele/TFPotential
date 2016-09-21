import tensorflow 			as 	tf
import numpy 				as 	np
import matplotlib.pyplot 	as 	plt
import datetime 			as 	time
import argumentparser 		as 	ap
import filefinder			as  ff
import neuralnetwork	 	as  nn



class TFPotential :
	def __init__(self) :
		self.argumentParser = ap.ArgumentParser()
		self.filefinder 	= ff.FileFinder(self.argumentParser)
		self.network 		= nn.NeuralNetwork()

		self.nLayers 	 = self.argumentParser.nLayers()
		self.nNodes  	 = self.argumentParser.nNodes()
		self.networkType = self.argumentParser.type()

		self.network.constructNetwork(inputs		= 1, 
						 			  nNodes		= self.nNodes,
						 			  nLayers		= self.nLayers, 
						 			  outputs		= 1, 
						 			  networkType	= None) 


	def __call__(self) :
		pass


tfpot = TFPotential()
