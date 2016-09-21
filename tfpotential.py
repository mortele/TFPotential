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

	def __call__(self) :
		return self.argumentParser()


tfpot = TFPotential()
