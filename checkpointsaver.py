import os
import tensorflow 	as 	tf


class CheckpointSaver :
	def __init__(self, system, save) :
		self.system 			= system
		self.save 				= save
		self.bestTestCost 		= None
		self.saveSkip			= 2
		self.index 				= 5
		self.checkpointNumber 	= 0
		self.firstEpoch			= True
		self.saverInitialized	= False
		self.saveDirectory, self.metaName = self.system.fileFinder.createSaveDirectory()

	def saveCheckpoint(self, epoch, testCost, session) :
		returnValue = False
		if self.save :
			if self.saverInitialized == False :
				self.saver 				= tf.train.Saver(max_to_keep=None)
				self.saverInitialized 	= True

			if testCost != -1 :
				if self.index >= 5 :
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

			if self.firstEpoch :
				self.firstEpoch = False
				with open(self.metaName, "w") as outFile :
					inputs 	= self.system.inputs
					layers 	= self.system.nLayers
					nodes 	= self.system.nNodes
					outputs = self.system.outputs
					nType 	= self.system.networkType
					outFile.write("%d %d %d %d %d %d %s" % (inputs, 
															layers, 
															nodes,
															outputs, 
															self.system.dataSize, 
															self.system.testSize,
															nType))

			with open(self.metaName, "a") as outFile :
				outFile.write("%d %.15g %.15g" % (epoch, 
												  self.system.networkTrainer.epochCost, 
												  testCost))
			return returnValue

	def loadCheckpoint(self, fileName, session) :
		if fileName != None :
			if self.saverInitialized == False :
				self.saverInitialized = True
				self.saver = tf.train.Saver(max_to_keep=None)
			self.saver.restore(self.system.networkTrainer.sess, fileName)
			return fileName
		else :
			return False


