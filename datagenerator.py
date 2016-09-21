import numpy 		as 	np


class DataGenerator :
	def __init__(self, a, b) :
		self.a = a
		self.b = b

	def setFunction(self, f) :
		self.function = function

	def generateData(self, n) :
		x = np.random.uniform(self.a, self.b, numberOfSamples)
		x = x.reshape([numberOfSamples, 1])
		y = self.function(x)
		return x, y