import neuralnet, numpy

LEFT = 0
RIGHT = 1
COMPLETE = 0
INCOMPLETE = 1

def Eisner(inputs, oracle):
	n = len(inputs)
	C = [[[[{'score': 0.0, 'arcs': set()} for i in range(2)] for j in range(2)] for k in range(n)] for l in range(n)]
	for m in range(1, n):
		for i in range(n - m):
			j = i + m
			bestarcs = set()
			for q in range(i, j):
				score = neuralnet.forwardpass(numpy.concatenate([inputs[j], inputs[i]]), oracle['weights'], oracle['biases'], neuralnet.VECTORIZEDNONLINEARITY)
				if C[i][q][RIGHT][COMPLETE]['score'] + C[q + 1][j][LEFT][COMPLETE]['score'] + score[2][0][0] > C[i][j][LEFT][INCOMPLETE]['score']:
					C[i][j][LEFT][INCOMPLETE]['score'] = C[i][q][RIGHT][COMPLETE]['score'] + C[q + 1][j][LEFT][COMPLETE]['score'] + score[2][0][0]
					bestarcs = C[i][q][RIGHT][COMPLETE]['arcs'].union(C[q + 1][j][LEFT][COMPLETE]['arcs'])
					bestarcs.add((j, i))
			C[i][j][LEFT][INCOMPLETE]['arcs'] = bestarcs
			bestarcs = set()
			for q in range(i, j):
				score = neuralnet.forwardpass(numpy.concatenate([inputs[i], inputs[j]]), oracle['weights'], oracle['biases'], neuralnet.VECTORIZEDNONLINEARITY)
				if C[i][q][RIGHT][COMPLETE]['score'] + C[q + 1][j][LEFT][COMPLETE]['score'] + score[2][0][0] > C[i][j][RIGHT][INCOMPLETE]['score']:
					C[i][j][RIGHT][INCOMPLETE]['score'] = C[i][q][RIGHT][COMPLETE]['score'] + C[q + 1][j][LEFT][COMPLETE]['score'] + score[2][0][0]
					bestarcs = C[i][q][RIGHT][COMPLETE]['arcs'].union(C[q + 1][j][LEFT][COMPLETE]['arcs'])
					bestarcs.add((i, j))
			C[i][j][RIGHT][INCOMPLETE]['arcs'] = bestarcs
			bestarcs = set()
			for q in range(i, j):
				if C[i][q][LEFT][INCOMPLETE]['score'] + C[q][j][LEFT][COMPLETE]['score'] > C[i][j][LEFT][COMPLETE]['score']:
					C[i][j][LEFT][COMPLETE]['score'] = C[i][q][LEFT][INCOMPLETE]['score'] + C[q][j][LEFT][COMPLETE]['score']
					bestarcs = C[i][q][LEFT][INCOMPLETE]['arcs'].union(C[q][j][LEFT][COMPLETE]['arcs'])
			C[i][j][LEFT][COMPLETE]['arcs'] = bestarcs
			bestarcs = set()
			for q in range(i, j):
				if C[i][q][RIGHT][COMPLETE]['score'] + C[q][j][RIGHT][INCOMPLETE]['score'] > C[i][j][RIGHT][COMPLETE]['score']:
					C[i][j][RIGHT][COMPLETE]['score'] = C[i][q][RIGHT][COMPLETE]['score'] + C[q][j][RIGHT][INCOMPLETE]['score']
					bestarcs = C[i][q][RIGHT][COMPLETE]['arcs'].union(C[q][j][RIGHT][INCOMPLETE]['arcs'])
			C[i][j][RIGHT][COMPLETE]['arcs'] = bestarcs
	return C[0][n - 1][RIGHT][COMPLETE]
