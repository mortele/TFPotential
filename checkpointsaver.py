import os
import tensorflow 	as 	tf


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

	def saveCheckpoint(self, epoch, testCost, session) :
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
						self.saveNetwork(self.checkpointNumber)
						self.checkpointNumber += 1
						self.index  = 0
					else :
						self.index += 1
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
					outFile.write("%d %d %d %d %d %d %s\n" % (inputs, 
															  layers, 
															  nodes,
															  outputs, 
															  self.system.dataSize, 
															  self.system.testSize,
															  nType))

			with open(self.metaName, "a") as outFile :
				outFile.write("%d %.15g %.15g\n" % (epoch, 
													self.system.networkTrainer.epochCost, 
													testCost))
			return returnValue

	def saveNetwork(self, checkpointNumber) :
		trainer = self.system.networkTrainer
		sess 	= trainer.sess
		var 	= tf.trainable_variables()
		sess.run(tf.initialize_all_variables())
		
		networkSaveFileName = 'network-%d' % checkpointNumber
		networkFile = os.path.join(self.saveDirectory, networkSaveFileName)
		with open(networkFile, 'w') as saveFile :
			inputs 	= self.system.inputs
			nLayers = self.system.nLayers
			nNodes 	= self.system.nNodes
			outputs = self.system.outputs
			saveFile.write('%d %d %d %d\n' % (	inputs,
												nLayers, 
												nNodes,
												outputs))
			for layer in xrange(self.system.nLayers) :
				w = sess.run([v.name for v in var if v.name == 'w%d:0' % layer])
				b = sess.run([v.name for v in var if v.name == 'b%d:0' % layer])

				for i in xrange(inputs if layer == 0 else nNodes) :
					for j in xrange(self.system.nNodes) :
						saveFile.write('%20.16f ' % w[0][i][j])
					saveFile.write('\n')
				for i in xrange(inputs if layer == 0 else nNodes) :
					saveFile.write('%20.16f ' % b[0][i])
				if layer != nLayers-1 :
					saveFile.write('\n')
		

				




	def loadCheckpoint(self, fileName, session) :
		if fileName != None :
			if self.saverInitialized == False :
				self.saverInitialized = True
				self.saver = tf.train.Saver(max_to_keep=None)
			self.saver.restore(self.system.networkTrainer.sess, fileName)
			return fileName
		else :
			return False

	def setSaveSkip(self, saveEach) :		
		self.saveSkip = saveEach













