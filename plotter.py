import matplotlib.pyplot 	as plt 
import numpy 				as np


class Plotter :
	def __init__(self, system) :
		self.system = system

	def plot(self) :
		self.plotFlag 			= self.system.argumentParser().plot
		self.plotProgressFlag 	= self.system.argumentParser().plotprogress
		self.plotErrorFlag		= self.system.argumentParser().ploterror
		self.plotAllFlag		= self.system.argumentParser().plotall

		self.plotFunction	(self.plotAllFlag or self.plotFlag)
		self.plotError 		(self.plotAllFlag or self.plotErrorFlag)
		self.plotProgress 	(self.plotAllFlag or self.plotProgressFlag)
		plt.show()

	def plotFunction(self, show=False) :
		if show :
			x, y = self.system.dataGenerator.generateData(1000)
			y_   = self.system.networkTrainer.sess.run(
								self.system.networkTrainer.prediction,
								feed_dict = {self.system.networkTrainer.x: x,
											 self.system.networkTrainer.y: y})
			plt.figure()
			plt.plot(x,y,'b--')
			plt.hold('on')
			plt.plot(x,y_,'r-')
			plt.legend(['V(r)','NN(r)'])
			plt.xlabel('r')
			plt.ylabel('V(r)')
			plt.title('Comparison of the potential and the approximating ANN.')


	def plotError(self, show=False) :
		if show :
			x, y = self.system.dataGenerator.generateData(1000)
			y_   = self.system.networkTrainer.sess.run(
								self.system.networkTrainer.prediction,
								feed_dict = {self.system.networkTrainer.x: x,
											 self.system.networkTrainer.y: y})
			err  = abs(y-y_) 
			plt.figure()
			plt.semilogy(x,err,'r-')
			plt.legend(['|V(r)-NN(r)|'])
			plt.xlabel('r')
			plt.ylabel('abs. network error(r)')
			plt.title('Absolute difference between the potential and the approximating ANN.')

	def plotProgress(self, show=False) :
		if show :
			if self.system.argumentParser().load != False :
				metaFile = self.system.fileFinder.loadMetaFile

				epoch 		= []
				epochCost 	= []
				testCost  	= []
				i = 0
				for line in open(metaFile, 'r') :
					if i != 0 :
						lineList = line.split()
						epoch.append	(int(lineList[0]))
						epochCost.append(float(lineList[1]))
						testCost.append (float(lineList[2]))
					i += 1

				for i in xrange(len(testCost)) :
					if testCost[i] == -1 :
						testCost[i] = np.nan;
					else :
						testCost[i] /= self.system.testSize

				fig, ax1 = plt.subplots()
				ax1.semilogy(epoch, epochCost, 'b-')
				ax1.set_xlabel('epoch #')
				ax1.set_ylabel('cost', color='b')
				for tl in ax1.get_yticklabels():
				    tl.set_color('b')

				ax2 = ax1.twinx()
				ax2.semilogy(epoch, testCost, 'r.')
				ax2.set_ylabel('cost / data size', color='r')
				for tl in ax2.get_yticklabels():
				    tl.set_color('r')









