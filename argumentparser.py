import argparse as arg

class ArgumentParser :
	def __init__(self, system) :
		self.system = system
		description = 'Fit potential energy surfaces using tensorflow neural networks.'
		self.parser = arg.ArgumentParser(description=description)
		self.parser.add_argument('epochs', help='The number of training epochs',
								 type=int, default=-1, nargs='?')
		self.parser.add_argument('--plot', help='Plot the network potential and the error',
								 default=False, action='store_true')
		self.parser.add_argument('--ploterror', help='Plot the error',
								 default=False, action='store_true')
		self.parser.add_argument('--plotprogress', help='Plot the cost function against epoch number',
								 default=False, action='store_true')
		self.parser.add_argument('--plotall', help='Plot all the things',
								 default=False, action='store_true')
		self.parser.add_argument('--load', help='Load graph from previous training file',
								 default=False, nargs='?')
		self.parser.add_argument('--save', help='Save the graph and training data to file',
								 default=False, action='store_true')
		self.parser.add_argument('--size', help='[# of hidden layers, # of neurons]',
								 type=int, nargs='+')
		self.parser.add_argument('--saveeach', help='How many epochs between each graph save',
								 type=int, default=10)
		self.parser.add_argument('--type', help='Which activation functions to use',
								 choices=['sigmoid','relu','relu-sigmoid'],
								 default=False, nargs='?')
		self.args = self.parser.parse_args()


	def __call__(self) :
		return self.args

	def nLayers(self) :
		if self().size != None and self().size != False :
			return self().size[0]
		else :
			return 5

	def nNodes(self) :
		if self().size != None and self().size != False :
			return self().size[1]
		else :
			return 5

	def type(self) :
		if self().type != None and self().type != False :
			return self().type
		else :
			return None
