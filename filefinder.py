import os
import sys
import shutil 
import datetime 		as time
import argumentparser 	as ap



class FileFinder :
	def __init__(self, system) :
		self.system			= system
		self.argumentParser = system.argumentParser
		self.trainingDir 	= 'TrainingData'

		# If the program was called with a --load flag but no file name, the 
		# load attribute of the parser is None.
		if self.argumentParser().load == None :
			self.findLastTrainingDirectory()
			self.findLastCheckpoint()
			self.loadFile     = os.path.join(self.trainingDir, self.lastCheckpoint)
			self.loadMetaFile = os.path.join(self.loadFile.split('/ckpt')[0], 'meta.dat')

		# If the program was called without a --load flag, the load attribute of
		# the parser defaults to False. So if parser.load is neither None nor 
		# False, we assume a filename was explicitly given.
		elif self.argumentParser().load not in {None, False}  :
			self.loadFile 		= self.argumentParser().load
			self.loadMetaFile 	= os.path.join(self.loadFile.split('/ckpt')[0], 'meta.dat')

		elif self.argumentParser().load == False :
			self.loadFile 		= None
			self.loadMetaFile 	= None

	def findLastTrainingDirectory(self) :
		dirList = os.listdir(self.trainingDir)
		# First, remove any directory which is not named as a date according to 
		# 'day.month-hour.minute.second'.
		N = len(dirList)
		j = 0
		for i in xrange(N) :
			if dirList[j].startswith('.') :
				dirList.pop(j);
			else :
				j = j + 1
		N = len(dirList)

		# Extract the month, day, hour, minute, second for each directory.
		lst  = [[0 for i in xrange(N)] for i in xrange(5)]
		for i in xrange(N) :
			lst[0][i] = int(dirList[i].split('.')[1].split('-')[0])
			lst[1][i] = int(dirList[i].split('.')[0])
			lst[2][i] = int(dirList[i].split('-')[1].split('.')[0])
			lst[3][i] = int(dirList[i].split('-')[1].split('.')[1])
			lst[4][i] = int(dirList[i].split('-')[1].split('.')[2])

		# Find the index of the directory list corresponding to the last date 
		# and time.
		index = -1
		for i in xrange(5) :
			m = max(lst[i][:])
			for j in xrange(N) :
				if not lst[i][j] == m :
					for k in xrange(i,5) :
						lst[k][j] = 0
				else :
					index = j
		self.lastTrainingDir = dirList[index]


	def findLastCheckpoint(self) :
		fileList = os.listdir(os.path.join(self.trainingDir, self.lastTrainingDir))
		N = len(fileList)
		j = 0
		for i in xrange(N) :
			if not fileList[j].startswith('ckpt') or \
				fileList[j].endswith('.meta') :
				fileList.pop(j)
			else :
				j = j + 1
		N 			= len(fileList)
		ckptNumbers = [int((fileList[i].split('-')[1]).split('.')[0]) for i in xrange(N)]
		maxCkpt 	= max(ckptNumbers)
		lastCheckpointName = 'ckpt-%d' % (maxCkpt)
		self.lastCheckpoint = os.path.join(self.lastTrainingDir, lastCheckpointName)

	def createSaveDirectory(self) :
		if self.system.argumentParser().save :
			now 				= time.datetime.now().strftime("%d.%m-%H.%M.%S")
			self.saveDirName 	= os.path.join(self.trainingDir, now)
			self.saveMetaName	= os.path.join(self.saveDirName, 'meta.dat')
			os.makedirs(self.saveDirName) 
			return self.saveDirName, self.saveMetaName
		else :
			return None, None





