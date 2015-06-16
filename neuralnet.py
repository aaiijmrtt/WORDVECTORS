import numpy, math

hadamard = numpy.vectorize(lambda x, y: x * y)

def feedforward(weights, inputs, biases, vectorizednonlinearity):
	return vectorizednonlinearity(numpy.add(numpy.dot(weights, inputs), biases))

def deltaoutput(activations, outputs, vectorizedgradient, vectorizedderivative):
	return hadamard(vectorizedgradient(activations, outputs), vectorizedderivative(activations))

def deltahidden(weights, activations, deltas, vectorizedderivative):
	return hadamard(numpy.dot(weights.transpose(), deltas), vectorizedderivative(activations))

def backpropagate(activations, deltas):
	return numpy.dot(deltas, activations.transpose()), deltas

def forwardpass(inputs, weights, biases, vectorizednonlinearity):
	activations = [inputs]
	for i in range(len(weights)):
		activations.append(feedforward(weights[i], activations[i], biases[i], vectorizednonlinearity))
	return activations

def backwardpass(activations, weights, biases, outputs, vectorizedgradient, vectorizedderivative):
	layers = len(weights)
	deltaweights = [numpy.zeros(weight.shape, dtype = float) for weight in weights]
	deltabiases = [numpy.zeros(bias.shape, dtype = float) for bias in biases]
	deltas = [None for i in range(layers)]
	deltas[layers - 1] = deltaoutput(activations[layers], outputs, vectorizedgradient, vectorizedderivative)
	for i in reversed(range(layers - 1)):
		deltas[i] = deltahidden(weights[i + 1], activations[i + 1], deltas[i + 1], vectorizedderivative)
	for i in range(layers):
		deltaweights[i], deltabiases[i] = backpropagate(activations[i], deltas[i])
	return deltaweights, deltabiases

def train(inputs, outputs, weights, biases, nonlinearity = lambda x: 1.0 / (1.0 + math.exp(-x)), gradient = lambda x, y: x - y, derivative = lambda x: x * (1 - x), alpha = 0.05):
	layers = len(weights)
	vectorizednonlinearity = numpy.vectorize(nonlinearity)
	vectorizedgradient = numpy.vectorize(gradient)
	vectorizedderivative = numpy.vectorize(derivative)
	for vectorsin, vectorsout in zip(inputs, outputs):
		activations = forwardpass(vectorsin, weights, biases, vectorizednonlinearity)
		deltaweights, deltabiases = backwardpass(activations, weights, biases, vectorsout, vectorizedgradient, vectorizedderivative)
		for i in range(layers):
			weights[i] = numpy.subtract(weights[i], alpha * deltaweights[i])
			biases[i] = numpy.subtract(biases[i], alpha * deltabiases[i])
	return weights, biases

def test(inputs, outputs, weights, biases, nonlinearity = lambda x: 1.0 / (1.0 + math.exp(-x)), cost = lambda x, y: (x - y) ** 2.0 / 2.0):
	layers = len(weights)
	costs = 0.0
	vectorizednonlinearity = numpy.vectorize(nonlinearity)
	vectorizedcost = numpy.vectorize(cost)
	for vectorsin, vectorsout in zip(inputs, outputs):
		activations = forwardpass(vectorsin, weights, biases, vectorizednonlinearity)
		costs += sum(math.fabs(element) for element in vectorizedcost(activations[layers], vectorsout))
	return (costs / len(inputs))
