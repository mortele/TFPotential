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
			xTe  = self.system.networkTrainer.xTest
			yTe  = self.system.networkTrainer.yTest
			yTe_ = self.system.networkTrainer.sess.run(
								self.system.networkTrainer.prediction,
								feed_dict = {self.system.networkTrainer.x: xTe,
											 self.system.networkTrainer.y: yTe})
			xTr  = self.system.networkTrainer.xTrain
			yTr  = self.system.networkTrainer.yTrain
			yTr_ = self.system.networkTrainer.sess.run(
								self.system.networkTrainer.prediction,
								feed_dict = {self.system.networkTrainer.x: xTr,
											 self.system.networkTrainer.y: yTr})											 
			plt.figure()
			plt.plot(xTr,yTr,'b--')
			plt.hold('on')
			plt.plot(xTr,yTr_,'r-')
			plt.legend(['f(r)','NN(r)'])
			plt.xlabel('r')
			plt.ylabel('function value')
			plt.title('Comparison of the training data and the approximating ANN.')
			
			plt.figure()
			plt.plot(xTe,yTe,'b--')
			plt.hold('on')
			plt.plot(xTe,yTe_,'r-')
			plt.legend(['f(r)','NN(r)'])
			plt.xlabel('r')
			plt.ylabel('function value')
			plt.title('Comparison of the test data and the approximating ANN.')


	def plotError(self, show=False) :
		if show :
			xTr  = self.system.networkTrainer.xTest
			yTr  = self.system.networkTrainer.yTest
			yTr_ = self.system.networkTrainer.sess.run(
								self.system.networkTrainer.prediction,
								feed_dict = {self.system.networkTrainer.x: xTr,
											 self.system.networkTrainer.y: yTr})
			xTe  = self.system.networkTrainer.xTrain
			yTe  = self.system.networkTrainer.yTrain
			yTe_ = self.system.networkTrainer.sess.run(
								self.system.networkTrainer.prediction,
								feed_dict = {self.system.networkTrainer.x: xTr,
											 self.system.networkTrainer.y: yTr})						
			
			errTr  = abs(yTr-yTr_) 
			errTe  = abs(yTe-yTe_)
			plt.figure()
			plt.semilogy(xTr,errTr,'r-')
			plt.hold('on')
			plt.semilogy(xTe,errTe,'b-')
			plt.legend(['|train(r)-NN(r)|', '|test(r)-NN(r)|'])
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









