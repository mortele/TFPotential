import argparse as arg

class ArgumentParser :
	def __init__(self) :
		description = 'Fit potential energy surfaces using tensorflow neural networks.'
		self.parser = arg.ArgumentParser(description=description)
		self.parser.add_argument('epochs', help='The number of training epochs',\
								 type=int, default=-1, nargs='?')
		self.parser.add_argument('--plot', help='Plot the network potential and the error',\
								 default=False, action='store_true')
		self.parser.add_argument('--load', help='Load graph from previous training file',\
								 default=False, nargs='?')
		self.parser.add_argument('--save', help='Save the graph and training data to file',\
								 default=False, action='store_true')
		self.args = self.parser.parse_args()

	def __call__(self) :
		return self.args