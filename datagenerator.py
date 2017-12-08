import sys
import numpy as np


class DataGenerator :
	def __init__(self, a, b) :
		self.a = a
		self.b = b
		self.generateFunction = self.generateLinspace

	def setGeneratorType(self, generatorType="function") :
		self.generatorType = generatorType

	def setFunction(self, function) :
		self.function = function

	def generateLinspace(self, n) :
		x = np.linspace(self.a, self.b, n, dtype=np.float32)
		x = x.reshape([n, 1])
		y = self.function(x)
		return x, y

	def VMCData(self, n) :
		x = []
		y = []
		with open('VMC_H2.dat', 'r') as inFile :
			numberOfLines = 0
			for line in inFile :
				line = line.split()
				x.append(float(line[0]))
				y.append(float(line[1]))
				numberOfLines += 1
				if numberOfLines >= n :
					break

		x = np.asarray(x).reshape([numberOfLines, 1])
		y = np.asarray(y).reshape([numberOfLines, 1])
		return x, y

	def generateData(self, n) :
		if self.generatorType == "function" :
			return self.generateLinspace(n) 
		elif self.generatorType == "VMC" :
			return self.VMCData(n)
		else :
			raise NameError("Invalid training data type '%s' in DataGenerator." % self.generatorType)

