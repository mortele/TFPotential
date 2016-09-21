# -*- coding: utf-8 -*-
import sys
import dill.source as ds


class Printer :
	def __init__(self, system) :
		self.system = system

	def printSetup(self) :
		t = self.system.network.networkType
		print " "
		print "Initializing network:"
		print "  ╠═ layers   ", self.system.nLayers
		print "  ╠═ neurons  ", self.system.nNodes
		print "  ╚═ type     ", t if t != None else "sigmoid"


	def printStart(self) :
		f = ds.getsource(self.system.function)
		f = f.split(':')[1].split('\n')[0].strip()
		print " "
		print "Training network:"
		print "  ╠═ function       ", f
		print "  ╠═ data set size  %g" % self.system.dataSize
		print "  ╠═ batch size     %g" % self.system.batchSize
		print "  ╚═ test set size  %g" % self.system.testSize
		print " "

	def printProgress(self, epoch, testCost=None) :
		c  = self.system.networkTrainer.epochCost
		n  = self.system.dataSize
		nt = self.system.testSize

		if epoch % 20 == 0 :
			print "%-16s %-16s %-16s %-16s %-16s" % ("Epoch", 
												"Cost", 
												"Cost/DataSize", 
												"Test Cost",
												"Test Cost/TestSize")
		print "%-16d %-16.8g %-16.8g" % (epoch, c, c/n),
		if testCost != None :
			print "%-16.8g %-16.8g" % (testCost, testCost/nt)
		else :
			print " "