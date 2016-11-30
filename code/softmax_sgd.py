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
		self.W = theano.shared(value = np.zeros([n_in, n_out], dtype = theano.config.floatX), name = "W", borrow = True)
		self.b = theano.shared(value = np.zeros(n_out, dtype = theano.config.floatX), name = "b", borrow = True)

		self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
		self.y_pred = T.argmax(self.p_y_given_x, axis = 1) #Need to add self, not p_pred, instead self.p_pred

		self.params = [self.W, self.b]
		self.input = input #Add according to tutorial

	def NLL(self, y):
		#Compute the cost function using cross entropy
		self.cost = -T.mean( T.log(self.p_y_given_x)[T.arange(y.shape[0]) , y])
		return self.cost

	def error(self, y):
		#Compute the error rate
		if y.ndim != self.y_pred.ndim:
			print "ERROR: The dimension is not equality!"
		if y.dtype != "int32" :
			print "ERROR: The type is not int64"
		return T.mean(T.neq(self.y_pred, y)) #not T.sum


def load_data(dataset = "mnist.pkl.gz"):
	path = os.path.join(os.path.split(__file__)[0], "..", "data", dataset)
	f = gzip.open(path, 'rb')
	train_set, valid_set, test_set = pickle.load(f)

	def shared_dataset(data_xy, borrow = True):
		data_x, data_y = data_xy
		shared_x = theano.shared(value = np.asarray(data_x , dtype = theano.config.floatX), borrow = borrow)
		shared_y = theano.shared(value = np.asarray(data_y, dtype = theano.config.floatX), borrow = borrow)
		#theano.shared(value = "a numpy matrix, which can set dtype")
		return shared_x, T.cast(shared_y, "int32") #This is int32, not int64

	train_x, train_y = shared_dataset(train_set) 
	valid_x, valid_y = shared_dataset(valid_set)
	test_x, test_y = shared_dataset(test_set)

	res = [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]
	return res

def sgd_optimization_minist(lr = 0.13, n_epochs = 1000, dataset = "mnist.pkl.gz", batch_size = 600):
	print "Loading data..."
	(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(dataset)
	train_batch_cnt = train_x.get_value(borrow = True).shape[0] // batch_size  #not use len(train_y), no this function
	valid_batch_cnt = valid_x.get_value(borrow = True).shape[0] // batch_size #add borrow = True
	test_batch_cnt = test_x.get_value(borrow = True).shape[0] // batch_size
	#Build model
	print "Building the model... "
	index = T.lscalar('index')
	x = T.matrix('x')  # data input
	y = T.ivector('y') #data output, not iscalar, because y is vector..
	sr = SoftmaxRegression(input = x, n_in = 784, n_out = 10)
	cost = sr.NLL(y)

	test_model = theano.function([index], sr.error(y), 
		givens={x: test_x[index * batch_size : (index+1) * batch_size], 
		y: test_y[index * batch_size : (index+1) * batch_size]})

	valid_model = theano.function([index], sr.error(y), 
		givens={x: valid_x[index * batch_size : (index+1) * batch_size], 
		y: valid_y[index * batch_size : (index+1) * batch_size]})

	dw = T.grad(cost = cost, wrt = sr.W)
	db = T.grad(cost = cost, wrt = sr.b)
	#updates = {sr.W : sr.W - lr * dw, sr.b : sr.b - lr * db} #Should be OrderedDict
	updates = [(sr.W, sr.W - lr * dw), (sr.b, sr.b - lr * db)]
	train_model = theano.function([index], cost,
		givens={x: train_x[index * batch_size : (index+1) * batch_size],
		y: train_y[index * batch_size : (index+1) * batch_size]},
		updates = updates)

	#Train model
	print "Training model..."
	epoch = 0
	while (epoch < n_epochs):
		#for i in T.arange(train_batch_cnt):
		#	loss = train_model(i)
		loss = [train_model(i) for i in range(train_batch_cnt)] #range not T.arange
		error = np.mean([valid_model(i) for i in range(valid_batch_cnt)]) #np.mean not T.mean
		print "epoch= ", epoch, " loss = ", loss[-1], " error= ", error
		epoch = epoch + 1


if __name__ == "__main__":
	sgd_optimization_minist()
