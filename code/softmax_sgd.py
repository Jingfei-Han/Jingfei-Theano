import theano
import theano.tensor as T
import gzip
import cPickle as pickle
import numpy as np
import timeit
import os

class SoftmaxRegression(object):

	def __init__(self, input, n_in, n_out):
		"""
		input: number of sample * n_in
		n_in: the number of input features
		n_out: the number of class
		"""
		self.W = theano.shared(value = np.zeros(n_in, n_out), name = "W", allow = True, dtype = theano.config.floatX)
		self.b = theano.shared(value = np.zeros(n_out), name = "b", allow = True, dype = theano.config.floatX)

		p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
		y_pred = T.argmax(p_y_given_x, axis = 1)

		self.params = [self.W, self.b]

	def NLL(self, y):
		#Compute the cost function using cross entropy
		self.cost = -T.mean(p_y_given_x[T.arange(input.shape[0]) , y])
		return self.cost

	def error(self, y):
		#Compute the error rate
		if y.ndim != y_pred.ndim:
			print "ERROR: The dimension is not equality!"
		if y.dtype != "int64" :
			print "ERROR: The type is not int64"
		return T.sum(T.neq(y_pred, y))


def load_data(dataset = "mnist.pkl.gz"):
	path = os.path.join(os.path.split(__file__)[0], "..", "data", dataset)
	f = gzip.open(path, 'rb')
	train_set, valid_set, test_set = pickle.load(f)

	def shared_dataset(data_xy, borrow = True):
		(data_x, data_y) = data_xy
		shared_x = theano.shared(value = data_x, borrow = borrow, dtype = theano.config.floatX)
		shared_y = theano.shared(value = data_y, borrow = borrow, dtype = theano.config.floatX)
		return shared_x, T.cast(shared_y, "int64")

	train_x, train_y = shared_dataset(train_set)
	valid_x, valid_y = shared_dataset(valid_set)
	test_x, test_y = shared_dataset(test_set)

	res = [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]
	return res

if __name__ == "__main__":
	load_data()
