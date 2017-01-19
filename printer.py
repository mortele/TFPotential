# -*- coding: utf-8 -*-
import sys
#import dill.source as ds


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
		#f = ds.getsource(self.system.function)
		#f = f.split(':')[1].split('\n')[0].strip()
		f 	= " "
		print " "
		print "Training network:"
		print "  ╠═ epochs         %-g" % self.system.numberOfEpochs
		print "  ╠═ function       %-s" % f
		print "  ╠═ data set size  %-g" % self.system.dataSize
		print "  ╠═ batch size     %-g" % self.system.batchSize
		print "  ╚═ test set size  %-g" % self.system.testSize
		print " "

	def printProgress(self, epoch, testCost=None, saved=None) :
		c  = self.system.networkTrainer.epochCost
		n  = self.system.dataSize
		nt = self.system.testSize

		if epoch % 30 == 0 :
			print "\n%-10s %-16s %-16s %-16s %-18s" % ("Epoch", 
													   "Cost", 
													   "Cost/DataSize", 
													   "Test Cost",
													   "Test Cost/TestSize")
			print "═"*(17*5-6+2)
		print "%-10d %-16.8g %-16.8g" % (epoch, c, c/n),
		if testCost not in {-1, None} :
			print "%-16.8g %-18.8g" % (testCost, testCost/nt),
			if saved != None :
				print "%s" % (" " if saved == False else "saved: "+saved)
			else :
				print " "
		else :
			print " "
		sys.stdout.flush()

	def printLoad(self, fileName) :
		if fileName != False :
			print "\nLoading graph: ", fileName