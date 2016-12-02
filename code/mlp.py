import numpy as np  
import theano
import theano.tensor as T 
from softmax_sgd import SoftmaxRegression, load_data

class HiddenLayer(object):

	def __init__(self, rng, input, n_in, n_hidden, W = None, b = None, activation = T.tanh):
		if W is None:  #When W is None, we need to initialize it!, not "W is not None" 
			W_value = np.asarray(
				rng.uniform(
					low = -np.sqrt(6 / (n_in + n_hidden)),
					high = np.sqrt(6 / (n_in + n_hidden)),
					size = (n_in, n_hidden)
					),
				dtype = theano.config.floatX)
			if activation == T.nnet.sigmoid: #Only adjust W
				W_value *= 4
			W = theano.shared(
				value = W_value, 
				name = 'W',
				borrow = True
				)
			
		if b is None:
			"""
			# b don't need random init, just let b is zero vector
			b = theano.shared(
				np.asarray(
					rng.uniform(
						low = -np.sqrt(6/(n_in + n_hidden)), 
						high = np.sqrt(6/(n_in + n_hidden)), 
						size = (n_hidden, )
						), 
					dtype= theano.config.floatX
					), 
				name = 'b',
				borrow = True
				)
			"""
			b = theano.shared(
				np.zeros((n_hidden,),
					dtype = theano.config.floatX
					),
				name = 'b',
				borrow = True)

		self.W = W
		self.b = b
		self.params = [self.W, self.b]

		lin_output = T.dot(input, self.W) + self.b
		"""
		#This is a bad implement
		#Must judge whether the activation is None
		if activation == None:
			self.output = lin_output
		else :
			self.output = activation()
		"""
		self.output = (lin_output if activation is None else activation(lin_output))


class MLP(object):
	def __init__(self, rng, input, n_in, n_hidden, n_out):

		self.HL = HiddenLayer(rng, input, n_in, n_hidden, activation = T.tanh)
		self.SR = SoftmaxRegression(self.HL.output, n_hidden, n_out)
		self.params = self.HL.params + self.SR.params  # Not [HL.params, SR.params], this is WRONG!
		self.NLL = self.SR.NLL
		self.error = self.SR.error

		#L1 regularization
		



def test_mlp(lr = 0.02, L1_reg = 0.00, L2_reg = 0.0001, n_epochs = 1000, dataset = "mnist.pkl.gz", batch_size = 20, n_hidden = 500):
	print "Loading data..."
	(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(dataset)
	train_batch_cnt = train_x.get_value(borrow = True).shape[0] // batch_size
	valid_batch_cnt = valid_x.get_value(borrow = True).shape[0] // batch_size
	test_batch_cnt = test_x.get_value(borrow = True).shape[0] // batch_size

	rng = np.random.RandomState(1234)

	print "Building the model..."
	index = T.lscalar('index')
	x = T.matrix('x')
	y = T.ivector('y')
	#mlp = MLP() # Need 5 arguments
	mlp = MLP(rng = rng, input = x, n_in = 784, n_hidden = n_hidden, n_out = 10)
	cost = mlp.NLL(y)
	dpara = [T.grad(cost, wrt = p) for p in mlp.params]
	
	updates = [(p, p - lr * dp) for (p, dp) in zip(mlp.params, dpara)]

	test_model = theano.function(
		inputs = [index], 
		outputs = mlp.error(y), #NOT error 
		givens = {x : test_x[index * batch_size : (index+1) * batch_size],
		y : test_y[index * batch_size : (index +1) * batch_size]})

	valid_model = theano.function(
		inputs = [index], 
		outputs = mlp.error(y),
		givens = {
		x : valid_x[index * batch_size : (index+1) * batch_size],
		y : valid_y[index * batch_size : (index + 1) * batch_size]
		})

	train_model = theano.function(
		[index], 
		cost,
		updates = updates,
		givens = {x : train_x[index * batch_size : (index+1) * batch_size],
		y : train_y[index * batch_size : (index + 1) * batch_size]})
	"""
	Cannot convert Type TensorType(float32, matrix) (of Variable Subtensor{int64:int64:}.0) 
	into Type TensorType(int32, vector). You can try to manually convert Subtensor{int64:int64:}.0 
	into a TensorType(int32, vector).

	The cause is train_model = theano.function(...givens ={... y : train_x...})this should be train_y
	"""

	print "Training model..."
	epoch = 0
	while (epoch < n_epochs):
		loss = [train_model(i) for i in range(train_batch_cnt)]
		#print train_batch_cnt, "Max loss: ", np.max(loss)
		error = np.mean([valid_model(i) for i in range(valid_batch_cnt)])
		print "epoch = ", epoch, " loss = ", loss[-1], " error = ", error
		epoch = epoch + 1


if __name__ == '__main__':
	test_mlp()
		

