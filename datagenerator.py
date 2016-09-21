import numpy 		as 	np


class DataGenerator :
	def __init__(self, a, b) :
		self.a = a
		self.b = b

	def setFunction(self, function) :
		self.function = function

	def generateData(self, n) :
		x = np.random.uniform(self.a, self.b, n)
		x = x.reshape([n, 1])
		y = self.function(x)
		return x, y