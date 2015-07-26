import neuralnet, numpy

SHIFT = -1
REDUCELEFT = 0
REDUCERIGHT = 1

def transition(stack, queue, arcs, dependencies):
	if len(stack) < 2:
		return (SHIFT, SHIFT, SHIFT)
	for dependency in dependencies:
		if stack[-1] == dependency[0] and stack[-2] == dependency[1]:
			return dependency
	for dependency in dependencies:
		if stack[-2] == dependency[0] and stack[-1] == dependency[1]:
			flag = True
			for dependence in dependencies:
				if dependence[0] == stack[-1] and dependence not in arcs:
					flag = False
			if flag:
				return dependency
	return (SHIFT, SHIFT, SHIFT)

def trainoracle(inputs, outputs, oracle, labels):
	vectorizednonlinearity = neuralnet.VECTORIZEDNONLINEARITY
	embeddingsize = inputs[0].shape[0]
	stack = [inputs[0]]
	queue = inputs[1: ]
	stackindices = [0]
	queueindices = range(1, len(inputs))
	arcs = list()
	ins = list()
	outs = list()
	while len(stack) > 1 or len(queue) > 0:
		besttransition = transition(stackindices, queueindices, arcs, outputs)
		if len(stack) > 1 and len(queue) > 0:
			ins.append(numpy.concatenate([stack[-2], stack[-1], queue[0]]))
		elif len(stack) > 1:
			ins.append(numpy.concatenate([stack[-2], stack[-1], numpy.zeros((embeddingsize, 1), dtype = float)]))
		else:
			ins.append(numpy.concatenate([numpy.zeros((embeddingsize, 1), dtype = float), stack[-1], queue[0]]))
		outs.append(numpy.zeros((2 * (labels + 1) + 1, 1), dtype = float))
		outs[-1][besttransition[2]][0] = 1.0
		if besttransition == (SHIFT, SHIFT, SHIFT):
			stack.append(queue.pop(0))
			stackindices.append(queueindices.pop(0))
		else:
			arcs.append(besttransition)
#			stack[stackindices.index(besttransition[0])] = neuralnet.forwardpass(vectorin, oracle['weights'], oracle['biases'], vectorizednonlinearity)[1]
			del stack[stackindices.index(besttransition[1])]
			del stackindices[stackindices.index(besttransition[1])]
	for i in range(len(inputs)):
		oracle['weights'], oracle['biases'] = neuralnet.train(ins, outs, oracle['weights'], oracle['biases'], alpha = [0.05, 0.05], gamma = [0.5, 0.5], history = i, hiddeninitializer = [numpy.zeros((embeddingsize, 1), dtype = float), numpy.zeros((2 * labels + 3, 1), dtype = float)])
	return oracle

def shiftreduce(inputs, oracle):
	vectorizednonlinearity = neuralnet.VECTORIZEDNONLINEARITY
	embeddingsize = inputs[0].shape[0]
	classes = oracle['biases'][1].shape[0]
	stack = [inputs[0]]
	queue = inputs[1: ]
	stackindices = [0]
	queueindices = range(1, len(inputs))
	arcs = list()
	hidden = [numpy.zeros((embeddingsize, 1), dtype = float), numpy.zeros((classes, 1), dtype = float)]
	while len(stack) > 1 or len(queue) > 0:
		bestscore = float("-inf")
		besttransition = None
		bestcombination = None
		bestlabel = None
		if len(stack) > 1:
			if len(queue) > 0:
				vectorin = numpy.concatenate([stack[-2], stack[-1], queue[0]])
			else:
				vectorin = numpy.concatenate([stack[-2], stack[-1], numpy.zeros((embeddingsize, 1), dtype = float)])
			activations = neuralnet.forwardpass(vectorin, oracle['weights'], oracle['biases'], vectorizednonlinearity, hidden)
			if numpy.max(activations[-1][0: -1]) > bestscore:
				bestscore = numpy.max(activations[2][0: -1])
				bestcombination = activations[1]
				bestlabel = numpy.argmax(activations[2][0: -1])
				besttransition = REDUCELEFT if bestlabel < classes // 2 else REDUCERIGHT
				besthidden = activations[1: ]
		if len(queue) > 0:
			if len(stack) > 1:
				vectorin = numpy.concatenate([stack[-2], stack[-1], queue[0]])
			else:
				vectorin = numpy.concatenate([numpy.zeros((embeddingsize, 1), dtype = float), stack[-1], queue[0]])
			activations = neuralnet.forwardpass(vectorin, oracle['weights'], oracle['biases'], vectorizednonlinearity, hidden)
			if activations[-1][-1][0] > bestscore:
				bestscore = activations[2][-1][0]
				bestcombination = None
				bestlabel = SHIFT
				besttransition = SHIFT
				besthidden = activations[1: ]
		hidden = besthidden
		if besttransition == SHIFT:
			stack.append(queue.pop(0))
			stackindices.append(queueindices.pop(0))
		else:
			arcs.append((stackindices[-1 - besttransition] + 1, stackindices[-2 + besttransition] + 1, bestlabel))
			del stack[-2 + besttransition]
			del stackindices[-2 + besttransition]
#			stack[-1] = bestcombination
	arcs.append((0, stackindices[0] + 1, REDUCERIGHT))
	return arcs
