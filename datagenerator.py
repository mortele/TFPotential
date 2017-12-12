import sys
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt



class DataGenerator :
	def __init__(self, a, b, system) :
		self.a = a
		self.b = b
		self.system = system
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
	
	def SW(self, nTrain, nTest=None) :
		
		def V(r) :
			return 10*7.04955627 * (0.6022245584 / (r*r*r*r) - 1.0 / r) * np.exp(1.0 / (r - 1.8))
		
		self.a = 1.8*0.45
		self.b = 1.8
		def temp(N) :
			n = int(np.floor(np.sqrt(N)))
			r1 = np.linspace(self.a, self.b-0.01, n)
			r2 = np.linspace(self.a, self.b-0.01, n)
			xTrain = np.zeros(shape=(n*n,2))
		
			for i in xrange(n) :
				for j in xrange(n) :
					ind = i*n+j
					xTrain[ind,0] = r1[i]
					xTrain[ind,1] = r2[j]
			yTrain = np.zeros(shape=(n*n,1))
			
			for i in xrange(n*n) :
				r1 = xTrain[i,0]
				r2 = xTrain[i,1]
			
				yTrain[i] = V(r1) + V(r2)
			return xTrain, yTrain, n
		
		
		xTrain, yTrain, nn = temp(nTrain)
		if nTest == None :
			self.system.dataSize = nn
			if self.system.batchSize > nn :
				self.system.batchSize = nn
			return xTrain, yTrain
		else :
			xTest, yTest, nn = temp(nTest)
			self.system.testSize = nn
			
			fig = plt.figure()
			ax = fig.add_subplot(111,projection='3d')
			ax.scatter(xTest[:,0], xTest[:,1], yTest,'r.')
			#plt.show()
			
			return xTrain, yTrain, xTest,yTest
			
			
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
			#plt.show()
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
	
	def fileData(self, n, fileName) :
		x = []
		y = []
		with open (fileName, 'r') as inFile :
			numberOfLines = 0
			for line in inFile : 
				line = line.split()
				x.append(float(line[0]))
				y.append(float(line[1]))
				numberOfLines += 1
				
		x = np.asarray(x).reshape([numberOfLines, 1])
		y = np.asarray(y).reshape([numberOfLines, 1])
		nn = numberOfLines
		
		xt = np.copy(x)
		yt = np.copy(y)
		
		print xt.shape
		if n < nn :
			toRemove = nn - n
			toRemove = np.random.choice(np.arange(len(xt)), toRemove, replace=False)
			xt = np.delete(xt, toRemove)
			yt = np.delete(yt, toRemove)
			xt = xt.reshape([n,1])
			yt = yt.reshape([n,1])
				
		self.system.testSize = nn
		
		print xt.shape
		print x.shape	
		self.system.dataSize = len(xt)
		return xt, yt, x, y
		
			
	def generateData(self, n, nTest=None) :
		if self.generatorType == "function" :
			return self.generateLinspace(n, nTest) 
		elif self.generatorType == "VMC" :
			return self.VMCData(n, nTest)
		elif self.generatorType == "random" or self.generatorType == "noise" :
			return self.noise(n, nTest)
		elif self.generatorType == "SW" :
			return self.SW(n, nTest)
		elif self.generatorType == "file" :
			return self.fileData(n, self.system.argumentParser().file)
		else :
			raise NameError("Invalid training data type '%s' in DataGenerator." % self.generatorType)

