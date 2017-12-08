import sys
import numpy as np
import matplotlib.pyplot as plt


class DataGenerator :
	def __init__(self, a, b) :
		self.a = a
		self.b = b
		self.generateFunction = self.generateLinspace

	def setGeneratorType(self, generatorType="function") :
		self.generatorType = generatorType

	def setFunction(self, function) :
		self.function = function
	
	def generateLinspace(self, nTrain, nTest) :
		x = np.linspace(self.a, self.b, nTrain, dtype=np.float32)
		x = x.reshape([nTrain, 1])
		y = self.function(x)
		if nTest == None :
			return x, y
		else :
			dx = (self.b - self.a)/float(3*nTest)
			xTest = np.linspace(self.a+dx, self.b+dx, nTest, dtype=np.float32)
			xTest = xTest.reshape([nTest, 1])		
			yTest = self.function(xTest)		
			return x, y, xTest, yTest
	
	def noise(self, nTrain, nTest=None) :
		
		def noiseFunction(r, n) :
			y = np.zeros(shape=r.shape)
			#y[int(np.round(n/3))] = 1
			y[2] = 1.0
			for i in xrange(30,100,2) :
				y[i] = np.random.normal(0,0.1/float(i/5.0)) 
				
			y = np.fft.irfft(y,axis=0)
			y = y[:n]
			
			return y/(np.max(y)-np.min(y))
	
		x = np.linspace(self.a, self.b, nTrain, dtype=np.float32)
		x = x.reshape([nTrain, 1])
		y = noiseFunction(x, nTrain)
		if nTest == None :
			return x, y
		else :
			yTest = noiseFunction(x, nTrain)
			
			plt.plot(x,y,'r-')
			plt.hold('on')
			plt.plot(x,yTest,'b--')
			plt.show()
			return x, y, x, yTest
			
	
	def VMCData(self, n, nTest=None) :
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
		if nTest == None :
			return x, y
		else :
			return x, y, x, y

	def generateData(self, n, nTest=None) :
		if self.generatorType == "function" :
			return self.generateLinspace(n, nTest) 
		elif self.generatorType == "VMC" :
			return self.VMCData(n, nTest)
		elif self.generatorType == "random" or self.generatorType == "noise" :
			return self.noise(n, nTest)
		else :
			raise NameError("Invalid training data type '%s' in DataGenerator." % self.generatorType)

