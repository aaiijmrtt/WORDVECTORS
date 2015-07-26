import numpy, math

hadamard = numpy.vectorize(lambda x, y: x * y) # hadamard
VECTORIZEDCOST = numpy.vectorize(lambda x, y: (x - y) ** 2 / 2.0) # half squared
VECTORIZEDGRADIENT = numpy.vectorize(lambda x, y: x - y) # half squared gradient
VECTORIZEDNONLINEARITY = numpy.vectorize(lambda x: math.tanh(x)) # hyperbolic tangent
VECTORIZEDDERIVATIVE = numpy.vectorize(lambda x: 1.0 - x ** 2) # hyperbolic tangent derivative
VECTORIZEDREGULARIZATION = numpy.vectorize(lambda x: 0.0) # no penalty

_VECTORIZEDNONLINEARITY = numpy.vectorize(lambda x: 1.0 / (1.0 + math.exp(-x))) # sigmoid
_VECTORIZEDDERIVATIVE = numpy.vectorize(lambda x: x * (1.0 - x)) # sigmoid derivative
_VECTORIZEDREGULARIZATION = numpy.vectorize(lambda x: 1.0 if x > 0.0 else - 1.0) # L1 penalty derivate
__VECTORIZEDREGULARIZATION = numpy.vectorize(lambda x: x) # L2 penalty derivative

ALPHA = [0.1, 0.05]
GAMMA = [0.45, 0.9]
LAMDA = 0.05
BATCH = 1
HISTORY = 1
HIDDENINITIALIZER = None
CLASSIFIER = False

def feedforward(weights, inputs, biases, vectorizednonlinearity):
	return vectorizednonlinearity(numpy.add(numpy.dot(weights, inputs), biases))

def deltaoutput(activations, outputs, vectorizedgradient, vectorizedderivative):
	return hadamard(vectorizedgradient(activations, outputs), vectorizedderivative(activations))

def deltahidden(weights, activations, deltas, vectorizedderivative):
	return hadamard(numpy.dot(weights.transpose(), deltas), vectorizedderivative(activations))

def backpropagate(activations, deltas):
	return numpy.dot(deltas, activations.transpose()), deltas

def forwardpass(inputs, weights, biases, vectorizednonlinearity, hiddenstates):
	activations = [None for i in range(len(weights) + 1)]
	activations[0] = inputs
	for i in range(len(weights)):
		if hiddenstates is None:
			activations[i + 1] = feedforward(weights[i], activations[i], biases[i], vectorizednonlinearity)
		else:
			activations[i + 1] = feedforward(weights[i], numpy.concatenate([activations[i], hiddenstates[i]]), biases[i], vectorizednonlinearity)
	return activations

def backwardpass(activations, weights, biases, outputs, vectorizedgradient, vectorizedderivative, hiddenstates):
	layers = len(weights)
	deltaweights = [None for i in range(layers)]
	deltabiases = [None for i in range(layers)]
	deltas = [None for i in range(layers)]
	deltas[layers - 1] = deltaoutput(activations[layers], outputs, vectorizedgradient, vectorizedderivative)
	for i in reversed(range(layers - 1)):
		if hiddenstates is None:
			deltas[i] = deltahidden(weights[i + 1], activations[i + 1], deltas[i + 1], vectorizedderivative)
		else:
			deltas[i] = deltahidden(weights[i + 1], numpy.concatenate([activations[i + 1], hiddenstates[i + 1]]), deltas[i + 1], vectorizedderivative)
	for i in range(layers):
		deltaweights[i], deltabiases[i] = backpropagate(activations[i], deltas[i]) if hiddenstates is None else backpropagate(numpy.concatenate([activations[i], hiddenstates[i]]), deltas[i][: activations[i + 1].shape[0]]) # check delta split
	return deltaweights, deltabiases

def train(inputs, outputs, weights, biases, alpha = ALPHA, gamma = GAMMA, lamda = LAMDA, vectorizednonlinearity = VECTORIZEDNONLINEARITY, vectorizedgradient = VECTORIZEDGRADIENT, vectorizedderivative = VECTORIZEDDERIVATIVE, vectorizedregularization = VECTORIZEDREGULARIZATION, batch = BATCH, history = HISTORY, hiddeninitializer = HIDDENINITIALIZER):
	velocityweights = [numpy.zeros(weight.shape, dtype = float) for weight in weights]
	velocitybiases = [numpy.zeros(bias.shape, dtype = float) for bias in biases]
	cumulativedeltaweights = [numpy.multiply(lamda, vectorizedregularization(weight)) for weight in weights]
	cumulativedeltabiases = [numpy.multiply(lamda, vectorizedregularization(bias)) for bias in biases]
	for count in range(len(inputs) - history + 1):
		hidden = hiddeninitializer
		for timestep in range(history):
			activations = forwardpass(inputs[count + timestep], weights, biases, vectorizednonlinearity, hidden)
			deltaweights, deltabiases = backwardpass(activations, weights, biases, outputs[count + timestep], vectorizedgradient, vectorizedderivative, hidden)
			for i in range(len(weights)):
				cumulativedeltaweights[i] = numpy.add(cumulativedeltaweights[i], deltaweights[i])
				cumulativedeltabiases[i] = numpy.add(cumulativedeltabiases[i], deltabiases[i])
			if hidden is not None:
				hidden = activations[1: ]
		if (count + 1) % batch == 0:
			velocityparameter = gamma[0] + (float(count) / len(inputs)) * (gamma[1] - gamma[0])
			weightparameter = alpha[0] + (float(count) / len(inputs)) * (alpha[1] - alpha[0])
			for i in range(len(weights)):
				velocityweights[i] = numpy.add(numpy.multiply(velocityparameter, velocityweights[i]), numpy.multiply(weightparameter, cumulativedeltaweights[i]))
				velocitybiases[i] = numpy.add(numpy.multiply(velocityparameter, velocitybiases[i]), numpy.multiply(weightparameter, cumulativedeltabiases[i]))
				weights[i] = numpy.subtract(weights[i], velocityweights[i])
				biases[i] = numpy.subtract(biases[i], velocitybiases[i])
				cumulativedeltaweights[i] = numpy.multiply(lamda, vectorizedregularization(weights[i]))
				cumulativedeltabiases[i] = numpy.multiply(lamda, vectorizedregularization(biases[i]))
	return weights, biases

def test(inputs, outputs, weights, biases, vectorizednonlinearity = VECTORIZEDNONLINEARITY, vectorizedcost = VECTORIZEDCOST, history = HISTORY, hiddeninitializer = HIDDENINITIALIZER, classifier = CLASSIFIER):
	costs = 0.0
	for count in range(len(inputs) - history + 1):
		hidden = hiddeninitializer
		for timestep in range(history):
			activations = forwardpass(inputs[count + timestep], weights, biases, vectorizednonlinearity, hidden)
			if hidden is not None:
				hidden = activations[1: ]
		if classifier:
			if outputs[count + history - 1][numpy.argmax(activations[len(weights)])][0] != 1.0:
				costs += 1.0
		else:
			costs += sum(math.fabs(element) for element in vectorizedcost(activations[len(weights)], outputs[count + history - 1]))
	return (costs / len(inputs))
