import tensorflow 			as 	tf
import numpy 				as 	np
import matplotlib.pyplot 	as 	plt
import datetime 			as 	time
import argumentparser 		as 	ap
import filefinder			as  ff



class TFPotential :
	def __init__(self) :
		self.argumentParser = ap.ArgumentParser()
		self.filefinder 	= ff.FileFinder(self.argumentParser)

	def __call__(self) :
		return self.argumentParser()


tfpot = TFPotential()
print tfpot.filefinder.loadFile