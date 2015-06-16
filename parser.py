from neuralnet import *

SHIFT = -1
REDUCELEFT = 0
REDUCERIGHT = 1

NONLINEARITY = lambda x: 1.0 / (1.0 + math.exp(-x))
NONLINEARITY_MAX = 1.0
NONLINEARITY_MIN = 0.0

def transition(stack, queue, arcs, dependencies):
	if len(stack) < 2:
		return SHIFT, 'SHIFT'
	for dependency in dependencies:
		if stack[-1] == dependency[0] and stack[-2] == dependency[1]:
			return REDUCELEFT, dependency[2]
	for dependency in dependencies:
		if stack[-2] == dependency[0] and stack[-1] == dependency[1]:
			flag = True
			for dependence in dependencies:
				if dependence[0] == stack[-1] and dependence not in arcs:
					flag = False
			if flag:
				return REDUCERIGHT, dependency[2]
	return SHIFT, 'SHIFT'

def trainoracles(inputs, outputs, oracleshift, oraclesreduceleft, oraclesreduceright):
	vectorizednonlinearity = numpy.vectorize(NONLINEARITY)
	embeddingsize = inputs[0].shape[0]
	stack = [inputs[0]]
	queue = inputs[1: ]
	stackindices = [0]
	queueindices = range(1, len(inputs))
	arcs = list()
	while len(stack) > 1 or len(queue) > 0:
		bestcombination = None
		besttransition, bestlabel = transition(stackindices, queueindices, arcs, outputs)
		if len(stack) > 1:
			vectorin = numpy.concatenate([stack[-2], stack[-1]])
			for oracle in oraclesreduceleft:
				if besttransition == REDUCELEFT and bestlabel == oracle['label']:
					oracle['weights'], oracle['biases'] = train([vectorin], [numpy.ones((1, 1), dtype = float)], oracle['weights'], oracle['biases'])
					bestcombination = forwardpass(vectorin, oracle['weights'], oracle['biases'], vectorizednonlinearity)[1]
				else:
					oracle['weights'], oracle['biases'] = train([vectorin], [numpy.zeros((1, 1), dtype = float)], oracle['weights'], oracle['biases'])
			for oracle in oraclesreduceright:
				if besttransition == REDUCERIGHT and bestlabel == oracle['label']:
					oracle['weights'], oracle['biases'] = train([vectorin], [numpy.ones((1, 1), dtype = float)], oracle['weights'], oracle['biases'])
					bestcombination = forwardpass(vectorin, oracle['weights'], oracle['biases'], vectorizednonlinearity)[1]
				else:
					oracle['weights'], oracle['biases'] = train([vectorin], [numpy.zeros((1, 1), dtype = float)], oracle['weights'], oracle['biases'])
		if len(queue) > 0:
			vectorin = numpy.concatenate([stack[-1], queue[0]])
			if besttransition == SHIFT and bestlabel == oracleshift['label']:
				oracleshift['weights'], oracleshift['biases'] = train([vectorin], [numpy.ones((1, 1), dtype = float)], oracleshift['weights'], oracleshift['biases'])
			else:
				oracleshift['weights'], oracleshift['biases'] = train([vectorin], [numpy.zeros((1, 1), dtype = float)], oracleshift['weights'], oracleshift['biases'])
		if bestcombination is not None:
			arcs.append((stackindices[-1 - besttransition], stackindices[-2 + besttransition], bestlabel))
			del stack[-2 + besttransition]
			del stackindices[-2 + besttransition]
			stack[-1] = bestcombination
		else:
			stack.append(queue.pop(0))
			stackindices.append(queueindices.pop(0))
	return oracleshift, oraclesreduceleft, oraclesreduceright

def shiftreduce(inputs, oracleshift, oraclesreduceleft, oraclesreduceright):
	vectorizednonlinearity = numpy.vectorize(NONLINEARITY)
	embeddingsize = inputs[0].shape[0]
	stack = [inputs[0]]
	queue = inputs[1: ]
	stackindices = [0]
	queueindices = range(1, len(inputs))
	arcs = list()
	while len(stack) > 1 or len(queue) > 0:
		bestscore = NONLINEARITY_MIN
		besttransition = None
		bestcombination = None
		bestlabel = None
		if len(stack) > 1:
			vectorin = numpy.concatenate([stack[-2], stack[-1]])
			for oracle in oraclesreduceleft:
				activations = forwardpass(vectorin, oracle['weights'], oracle['biases'], vectorizednonlinearity)
				if activations[2][0, 0] > bestscore:
					bestscore = activations[2][0, 0]
					besttransition = REDUCELEFT
					bestcombination = activations[1]
					bestlabel = oracle['label']
			for oracle in oraclesreduceright:
				activations = forwardpass(vectorin, oracle['weights'], oracle['biases'], vectorizednonlinearity)
				if activations[2][0, 0] > bestscore:
					bestscore = activations[2][0, 0]
					besttransition = REDUCERIGHT
					bestcombination = activations[1]
					bestlabel = oracle['label']
		if len(queue) > 0:
			vectorin = numpy.concatenate([stack[-1], queue[0]])
			activations = forwardpass(vectorin, oracleshift['weights'], oracleshift['biases'], vectorizednonlinearity)
			if activations[2][0, 0] > bestscore:
				bestscore = activations[2][0, 0]
				besttransition = SHIFT
				bestcombination = None
				bestlabel = oracleshift['label']
		if bestcombination is not None:
			arcs.append((stackindices[-1 - besttransition], stackindices[-2 + besttransition], bestlabel))
			del stack[-2 + besttransition]
			del stackindices[-2 + besttransition]
			stack[-1] = bestcombination
		else:
			stack.append(queue.pop(0))
			stackindices.append(queueindices.pop(0))
	return arcs
