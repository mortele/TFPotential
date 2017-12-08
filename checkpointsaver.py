import os
import tensorflow 	as 	tf
from math import floor
import numpy as np

class CheckpointSaver :
	def __init__(self, system, save) :
		self.system 			= system
		self.save 				= save
		self.bestTestCost 		= None
		self.saveSkip			= 2
		self.index 				= 0
		self.checkpointNumber 	= 0
		self.firstEpoch			= True
		self.saverInitialized	= False
		self.saveDirectory, self.metaName = self.system.fileFinder.createSaveDirectory()
		self.setSaveSkip(self.system.argumentParser().saveeach)
		self.writerInitialized = False


	def saveCheckpoint(self, epoch, testCost, session) :
		if not self.writerInitialized :
			self.writer = tf.summary.FileWriter('/tmp/tensorflow/', graph=tf.get_default_graph())
			self.writerInitialized = True

		returnValue = False
		if self.save :
			if self.saverInitialized == False :
				self.saver 				= tf.train.Saver(max_to_keep=None)
				self.saverInitialized 	= True

			if testCost != -1 :
				if self.index >= self.saveSkip :
					if (self.bestTestCost == None) or (testCost < self.bestTestCost) :
						self.bestTestCost = testCost
						self.saver.save(session,
										os.path.join(self.saveDirectory, 'ckpt'),
										global_step=self.checkpointNumber)
						returnValue = "ckpt-"+str(self.checkpointNumber)
						self.checkpointNumber += 1
						
						self.index  = 0
					else :
						self.index += 1
				else :
					self.index += 1
			else :
				self.index += 1
				
			if epoch < 20 :
				self.saveNetworkOutput(session, epoch)
			elif epoch % 5 == 0 :
				self.saveNetworkOutput(session, epoch)

			if self.firstEpoch :
				self.firstEpoch = False
				with open(self.metaName, "w") as outFile :
					inputs 	= self.system.inputs
					layers 	= self.system.nLayers
					nodes 	= self.system.nNodes
					outputs = self.system.outputs
					nType 	= self.system.networkType
					outFile.write("%d %d %d %d %d %d %s\n" % (inputs, 
															  layers, 
															  nodes,
															  outputs, 
															  self.system.dataSize, 
															  self.system.testSize,
															  nType))
				with open(os.path.join(self.saveDirectory,'trainSet.txt'), 'w') as outFile :
					xTrain = self.system.networkTrainer.xTrain
					yTrain = self.system.networkTrainer.yTrain
					for i in xrange(len(xTrain)) :
						outFile.write('%30.20g %30.20g\n' % (xTrain[i], yTrain[i]))
				
				with open(os.path.join(self.saveDirectory,'testSet.txt'), 'w') as outFile :
					xTest  = self.system.networkTrainer.xTest
					yTest  = self.system.networkTrainer.yTest
					for i in xrange(len(xTest)) :
						outFile.write('%30.20g %30.20g\n' % (xTest[i], yTest[i]))

			with open(self.metaName, "a") as outFile :			
				#tf.summary.histogram(name, variable)       
				outFile.write("%d %.15g %.15g\n" % (epoch, 
													self.system.networkTrainer.epochCost, 
													testCost))			
			return returnValue

	def saveNetwork(self, session) :
		returnValue = False
		if self.save :
			returnValue = True
			trainer = self.system.networkTrainer
			sess 	= session
			var 	= tf.trainable_variables()
			
			networkSaveFileName = 'network'
			networkFile = os.path.join(self.saveDirectory, networkSaveFileName)
			#networkFile = os.path.join("/Users/morten/Documents/Master/TFPotential/C++Test", networkSaveFileName)
			with open(networkFile, 'w') as saveFile :
				inputs 	= self.system.inputs
				"""
				nLayers = self.system.nLayers
				nNodes 	= self.system.nNodes
				outputs = self.system.outputs
				saveFile.write('%d %d %d %d\n' % (	inputs,
													nLayers, 
													nNodes,
													outputs))
				
				for layer in xrange(self.system.nLayers+2) :
					w = sess.run([v.name for v in var if v.name == 'w%d:0' % layer])
					b = sess.run([v.name for v in var if v.name == 'b%d:0' % layer])

					iLimit = nNodes
					jLimit = nNodes
					if layer == 0 : iLimit = inputs
					if layer == nLayers+1 : jLimit = outputs

					for i in xrange(iLimit) :
						for j in xrange(jLimit) :
							saveFile.write('%20.16f ' % w[0][i][j])
						saveFile.write('\n')
					for i in xrange(min(nNodes, jLimit)) :
						saveFile.write('%20.16f ' % b[0][i])

					saveFile.write('\n')
				print("\n\nSaved network to %s\n" %networkFile)
				"""
		return returnValue			




	def loadCheckpoint(self, fileName, session) :
		if fileName != None :
			if self.saverInitialized == False :
				self.saverInitialized = True
				self.saver = tf.train.Saver(max_to_keep=None)
			self.saver.restore(self.system.networkTrainer.sess, fileName)
			self.loadTrainingSet()
			return fileName
		else :
			return False

	def setSaveSkip(self, saveEach) :		
		self.saveSkip = saveEach

	def saveNetworkOutput(self, session, epoch) :
		with open(os.path.join(self.saveDirectory, 'yTest.txt'), 'a') as outFile :
			y_ = session.run(self.system.networkTrainer.prediction,
								feed_dict ={self.system.networkTrainer.x : self.system.networkTrainer.xTest,
											self.system.networkTrainer.y : self.system.networkTrainer.yTest}) 
			outFile.write('%10d ' % epoch)
			if len(y_) > 1000 :
				skip = int(floor(len(y_)/1000))
				
			for i in xrange(0,len(y_),skip) :
				outFile.write('%30.20g ' % y_[i]);
			outFile.write('\n');
		
		with open(os.path.join(self.saveDirectory, 'yTrain.txt'), 'a') as outFile :
			y_ = session.run(self.system.networkTrainer.prediction,
								feed_dict ={self.system.networkTrainer.x : self.system.networkTrainer.xTrain,
											self.system.networkTrainer.y : self.system.networkTrainer.yTrain}) 
			outFile.write('%10d ' % epoch)
			if len(y_) > 1000 :
				skip = int(floor(len(y_)/1000))
			for i in xrange(0,len(y_),skip) :
				outFile.write('%30.20g ' % y_[i]);
			outFile.write('\n');


	def loadTrainingSet(self) :
		trainingDir 	= self.system.fileFinder.trainingDir
		lastTrainingDir = self.system.fileFinder.lastTrainingDir
		
		inFileName = os.path.join(os.path.join(trainingDir, lastTrainingDir), 'testSet.txt')
		N = self.file_len(inFileName)
		with open(inFileName, 'r') as inFile :
			xTest = np.zeros(shape=(N,1))
			yTest = np.zeros(shape=(N,1))
			for i in xrange(N) :
				line = inFile.readline().split()
				xTest[i] = float(line[0])
				yTest[i] = float(line[1])
				
		inFileName = os.path.join(os.path.join(trainingDir, lastTrainingDir), 'trainSet.txt')
		M = self.file_len(inFileName)
		with open(inFileName, 'r') as inFile :
			xTrain = np.zeros(shape=(N,1))
			yTrain = np.zeros(shape=(N,1))
			for i in xrange(N) :
				line = inFile.readline().split()
				xTrain[i] = float(line[0])
				yTrain[i] = float(line[1])
		
		self.system.networkTrainer.xTest = xTest
		self.system.networkTrainer.yTest = yTest
		
		self.system.networkTrainer.xTrain = xTrain
		self.system.networkTrainer.yTrain = yTrain
		
		self.system.dataSize = N
		self.system.testSize = M

	def file_len(self, fname):
		with open(fname) as f:
		    for i, l in enumerate(f):
		        pass
		return i + 1






