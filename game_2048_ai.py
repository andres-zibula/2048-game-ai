"""
Author:	Andres Zibula
Source:	https://github.com/andres-zibula/2048-game-ai
"""

import numpy as np
import random, os, uuid, math, time
from pyevolve import G1DList, GSimpleGA, Initializators, Mutators, Crossovers, Selectors
from pybrain.structure import FeedForwardNetwork, LinearLayer, TanhLayer, FullConnection

class Game2048AI(object):

	"""
	This is a implementation of an AI for the game 2048, currently it is using
	a neural network with a genetic algorithm for weight balancing, but since the
	results of this approach are not very good I am planning to add another approach
	using minimax or Q-learning
	"""
	
	def __init__(self, technique = "neural-network"):
		"""
		Initialize 
		
		Args:
		    technique (str, optional): The technique to use, default is neural-network
		"""

		if(technique == "neural-network"):
			#point the function getNextMove to getNextMoveNN
			self.getNextMove = self.getNextMoveNN

			self.nnInputNumber = 16
			self.nnHiddenNumber = 8
			self.nnOutNumber = 4

			#build the network
			self.initializeNN()

		#map 0 => 'right', 1 => 'down', 2 => 'left', 3 => 'up'
		self.slideTo = ('right', 'down', 'left', 'up')

	def initializeNN(self):
		"""
		build the network, for details go to the PyBrain manual
		"""
		self.network = FeedForwardNetwork()
		inLayer = LinearLayer(self.nnInputNumber)
		hiddenLayer = TanhLayer(self.nnHiddenNumber)
		outLayer = TanhLayer(self.nnOutNumber)
		
		self.network.addInputModule(inLayer)
		self.network.addModule(hiddenLayer)
		self.network.addOutputModule(outLayer)
		
		in_to_hidden = FullConnection(inLayer, hiddenLayer)
		hidden_to_out = FullConnection(hiddenLayer, outLayer)
		
		self.network.addConnection(in_to_hidden)
		self.network.addConnection(hidden_to_out)
		self.network.sortModules()

		#trained data, when training this is replaced
		data = [0.9425114397050811, -0.7948482757192381, 1.4178887806433025, 0.5829691091560636, -2.628075360781522, -1.7463737884748085, 0.09482633161672593, -0.05428818148642878, -2.2343024799789504, 2.664848143017821, -0.48502740861056637, 1.1256335407763371, -2.5542373021490734, 1.6637783890620415, 0.3017496685566927, 0.061613932183112397, -0.9537204390518159, -2.0896025919675596, -0.6219094487265657, -2.825988755304663, -2.2190456481083665, -0.7718953048375283, 1.264078926027115, 0.6892868949321, 2.0763290222581086, 2.890642129272762, -2.010217973351808, -0.3608718143461589, -2.43212392498987, -0.2718533737120463, 1.7059467696159558, 2.9680993656888397, 2.0071497635882167, 1.5240733704133813, -0.042841096506258225, -1.300740331625506, -1.286619945748014, -0.08210947017567705, -2.015699644592194, 2.2774544588313868, -1.6628056739105135, -0.38286597626730967, 2.519760565185713, -1.2069335635015537, -3.0, -1.983212696494711, 1.232995567653151, 0.025921886830332586, 0.19277810432262843, -0.25700163161472567, 0.964866782521515, -0.07961979767684646, 0.10041515284094693, -0.3258317903373267, 2.1071667353967776, 2.1294883579864656, 0.8912237441374167, 2.661028254731516, -1.9151996827668565, 1.1864787281580833, 0.10690593082779642, -2.5076451181287314, -1.4410434045993692, -1.0915377420502157, -1.4764230546704793, -0.23246432936510075, 0.03329537224639667, -2.790891284047849, -2.612408189062002, -0.39257750183015816, -0.23889088681978787, -1.9715054516361772, 1.3672904955740055, -2.700472549134502, 0.9283680811045745, -0.4848461954453902, -0.615159901883203, 1.5701219958802786, 0.400776328926546, 0.788068749150511, 2.704549164836843, -0.8251662437138068, -2.1179724770671484, -1.0182294413888182, -2.5020478142767955, -1.3931215850505687, -1.8379077101671175, -1.0685241187631023, -0.12929909458472189, 2.970932353605681, -1.4276467099378929, 2.144598752944783, 2.3449300983133394, 3.0, 1.6626486476821274, -0.5495533592987463, 2.5023116015327727, 0.48876931529671963, -0.9666755550061019, 0.4061780468109273, 0.4962370039950219, -3.0, -0.7989387530425076, -0.07011873549904632, -0.049756440609789176, 2.0142966233190895, 2.302207259133332, 1.0368222618442253, 1.5584744501441836, -1.385378940803622, 1.6431517631095973, -2.8754073722553417, -1.1126319637322983, 0.6059908131992557, -2.219953457414824, 2.5276898458393067, -1.6847951674801966, -0.7958425465800296, 1.7056375137405881, 0.23711578697824853, 2.286260350709026, -0.5794662633919748, -1.133892418870305, -0.1977451837264872, 0.5607329538904269, -2.561240300383358, -2.7216062701741306, -0.07833729737059558, -0.09101083708557844, 1.5860088832666364, -2.784095792102007, -0.4487926033248262, 2.8555561065065858, 2.207046614961932, 2.6001645287015824, -0.11015106011435716, -0.11141221091145637, 2.281511368401702, -1.324513001939784, 0.13901521014596785, 0.6802844954973928, 1.145856563901618, 0.28914653916551974, -0.7915226308859795, -1.121621711031553, -0.5098635356815935, -2.9424096830173836, 0.20195667982270749, 2.3933163344341626, -1.7118756732575915, -0.5773687816554275, -0.15147565141354624, 0.7227488513575031, 3.0, -1.6745750969941613, -2.6227079605589942, -1.1238033687084736, -1.7071547799370717, -2.088724059716908, 0.45694434380355564]
		weights = np.array(data)
		self.network._setParameters(weights)

	def play(self):
		"""
		Simulate a game
		"""

		currentIteration = 0
		matrix = self.newMatrix()
		score = 0

		while (not self.isGameOver(matrix)):

			oldMatrix = np.copy(matrix)
			direction = self.slideTo[self.getNextMove(matrix)]
			matrix, score = self.slideMatrix(matrix, direction, score)

			#if the matrix is the same as the previous one we choose a random action, until
			#the matrices are not the same, this is a workaround for the problem of getting stuck
			#because the inputs of the NN are the same, and the algorithm will keep trying with the same output
			if np.array_equal(matrix, oldMatrix):
				r = random.randrange(0, 4)
				matrix, score = self.slideMatrix(matrix, self.slideTo[r], score)

			self.printUI(matrix, score)
			currentIteration += 1
			time.sleep(0.3)

	def train(self):
		"""
		This function is for training the neural network
		"""
		executionId = str(uuid.uuid4()).replace('-','')
		statsFilename = "./stats/" + executionId + ".txt"
		if not os.path.exists(os.path.dirname(statsFilename)):
			os.makedirs(os.path.dirname(statsFilename))

		for iterations in xrange(1000):
			#initialize the genome type
			genome = G1DList.G1DList(self.nnInputNumber*self.nnHiddenNumber + self.nnHiddenNumber*self.nnOutNumber)
			genome.setParams(rangemin=-3.0, rangemax=3.0)
			genome.initializator.set(Initializators.G1DListInitializatorReal)
			genome.mutator.set(Mutators.G1DListMutatorRealGaussian)
			genome.crossover.set(Crossovers.G1DListCrossoverTwoPoint)
			genome.evaluator.set(self.scoreEval)
	
			#initialize PyEvolve genetic algorithm engine
			ga = GSimpleGA.GSimpleGA(genome)
			ga.setMutationRate(0.05)
			ga.selector.set(Selectors.GRankSelector)
			ga.setElitism(True)
			ga.setPopulationSize(200)
			ga.initialize()
			
			#we run up to generation 300
			for _ in range(300):
				#print for debug info
				print(ga.getCurrentGeneration())
				
				ga.step()

				#write the best genome to a file
				fh = open(statsFilename, "a")
				fh.write(str(ga.bestIndividual().score) + ";" + str(ga.bestIndividual().genomeList) + "\n")
				fh.close
	
			print(ga.bestIndividual().score)

	def scoreEval(self, chromosome):
		"""
		This function is called by the genetic algorithm engine in each generation step
		to evaluate the genome
		
		Args:
		    chromosome (G1DList): PyEvolve's individual container
		
		Returns:
		    Int: The score of the genome
		"""

		#replace the weights of the neural network
		weights = np.array(chromosome.genomeList)
		self.network._setParameters(weights)
		scores = []

		#we iterate with the same weights 20 times (20 different 2048 new game states), and
		#then return the minimun score
		for _ in range(20):

			currentIteration = 0
			score = 0
			matrix = self.newMatrix()
	
			while (not self.isGameOver(matrix)):

				oldMatrix = np.copy(matrix)
				direction = self.slideTo[self.getNextMove(matrix)]
				matrix, score = self.slideMatrix(matrix, direction, score)

				#if the matrix is the same as the previous one we choose a random action, until
				#the matrices are not the same, this is a workaround for the problem of getting stuck
				#because the inputs of the NN are the same, and the algorithm will keep trying with the same output
				if np.array_equal(matrix, oldMatrix):
					r = random.randrange(0, 4)
					matrix, score = self.slideMatrix(matrix, self.slideTo[r], score)

				currentIteration += 1
			scores.append(score)

		return min(scores)

	def printMatrix(self, matrix):
		"""
		Print the matrix
		
		Args:
		    matrix (Numpy Array): The numbers matrix
		"""
		print("+" + "-"*4 + "+" + "-"*4 + "+" + "-"*4 + "+" + "-"*4 + "+")
		print('|{0: ^4}|{1: ^4}|{2: ^4}|{3: ^4}|'.format(matrix[0][0], matrix[1][0], matrix[2][0], matrix[3][0]))
		print("|" + "-"*19 + "|")
		print('|{0: ^4}|{1: ^4}|{2: ^4}|{3: ^4}|'.format(matrix[0][1], matrix[1][1], matrix[2][1], matrix[3][1]))
		print("|" + "-"*19 + "|")
		print('|{0: ^4}|{1: ^4}|{2: ^4}|{3: ^4}|'.format(matrix[0][2], matrix[1][2], matrix[2][2], matrix[3][2]))
		print("|" + "-"*19 + "|")
		print('|{0: ^4}|{1: ^4}|{2: ^4}|{3: ^4}|'.format(matrix[0][3], matrix[1][3], matrix[2][3], matrix[3][3]))
		print("+" + "-"*4 + "+" + "-"*4 + "+" + "-"*4 + "+" + "-"*4 + "+")

	def printScore(self, score):
		"""
		Print the score
		
		Args:
		    score (Int): Score of the game until now
		"""
		print("Score: " + str(score))

	def printUI(self, matrix, score):
		"""
		Print the game interface
		
		Args:
		    matrix (Numpy Array): The numbers matrix
		    score (Int): Score of the game until now
		"""
		print(chr(27) + "[2J")
		print("")
		self.printMatrix(matrix)
		print("")
		self.printScore(score)

	def newMatrix(self):
		"""
		Initializes a new game state
		
		Returns:
		    Numpy Array: The numbers matrix
		"""
		matrix = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
		matrix = self.putRandomNumber(matrix)
		matrix = self.putRandomNumber(matrix)

		return matrix

	def getNextMove(self, matrix):
		"""
		Returns the best move for the current game state

		This function is overrided in the constructor with the technique choosed.
		
		Args:
		    matrix (Numpy Array): The numbers matrix

		Returns:
		    Int: An integer in the range [0,3], 0 => 'right', 1 => 'down', 2 => 'left', 3 => 'up'
		"""
		pass

	def getNextMoveNN(self, matrix):
		"""
		Returns the best move for the current game state
		
		Args:
		    matrix (Numpy Array): The numbers matrix

		Returns:
		    Int: An integer in the range [0,3], 0 => 'right', 1 => 'down', 2 => 'left', 3 => 'up'
		"""
		return np.argmax(self.network.activate(self.getNNInputs(matrix)))

	def getNNInputs(self, matrix):
		"""
		Returns the inputs for the NN for the current matrix, one input per cell
		An input is the log base 2 of the number in the cell divided by the log base 2 of
		the max number in the matrix

		Args:
		    matrix (Numpy Array): The numbers matrix
		
		Returns:
		    List: List of numbers for the inputs
		"""

		maxNumber = math.log(np.amax(matrix), 2)
		nnInputs = []

		for col in matrix:
			for n in col:
				if n == 0:
					nnInputs.append(0)
				else:
					nnInputs.append(math.log(n,2)/float(maxNumber))
				

		#nnInputs.append(1.0) #bias
		return nnInputs

	def slideMatrix(self, matrix, direction, score):
		"""
		Change the game state sliding in "direction"

		My algorithm works this way: I rotate the board x times so I can apply
		the slide down algorithm, and then I rotate it as it was before
		
		Args:
		    matrix (Numpy Array): The numbers matrix
		    direction (str): The direction to slide
		    score (Int): Score of the game until now
		
		Returns:
		    Tuple: matrix, score; the matrix after the slide and the new score
		"""

		if direction == "right":
			matrix = np.rot90(matrix, 1)
		elif direction == "left":
			matrix = np.rot90(matrix, -1)
		elif direction == "up":
			matrix = np.rot90(matrix, 2)
		#else is "down"

		movedAtLeastOneCell = False

		for col in matrix:
			lastMergeIndex = None

			for i in range(2, -1, -1):
				if col[i] == 0:
					continue

				j = i+1
				while (col[j] == 0) and (j < 3):
					j += 1

				if (col[i] == col[j]) and (lastMergeIndex != j):
					score += col[j]*2
					col[j] = col[j]*2
					col[i] = 0
					lastMergeIndex = j
					movedAtLeastOneCell = True
				elif col[j] == 0:
					col[j] = col[i]
					col[i] = 0
					movedAtLeastOneCell = True
				elif j-1 != i:
					col[j-1] = col[i]
					col[i] = 0
					movedAtLeastOneCell = True

		if movedAtLeastOneCell:
			matrix = self.putRandomNumber(matrix)

		if direction == "right":
			matrix = np.rot90(matrix, -1)
		elif direction == "left":
			matrix = np.rot90(matrix, 1)
		elif direction == "up":
			matrix = np.rot90(matrix, 2)

		return matrix, score

	def isAtLeastOneEmptyCell(self, matrix):
		"""
		Return True if the board has at leat one cell empty
		
		Args:
		    matrix (Numpy Array): The numbers matrix
		
		Returns:
		    Bool: True if the board has at leat one cell empty, False otherwise
		"""

		for col in matrix:
			for n in col:
				if n == 0:
					return True

		return False

	def putRandomNumber(self, matrix):
		"""
		Places a 2 or a 4 in the board
		
		Args:
		    matrix (Numpy Array): The numbers matrix
		
		Returns:
		    Numpy Array: The numbers matrix
		"""

		n = 4
		if random.uniform(0.0, 1.0) < 0.9:
			n = 2

		numberPlaced = False

		while not numberPlaced:
			x = random.randrange(0, 4)
			y = random.randrange(0, 4)

			if matrix[x][y] == 0:
				matrix[x][y] = n
				numberPlaced = True

		return matrix

	def isGameOver(self, matrix):
		"""
		Returns True if the matrix has no more valid moves
		
		Args:
		    matrix (Numpy Array): The numbers matrix
		
		Returns:
		    Bool: True if game is over, False otherwise
		"""

		#if the board has an empty cell game is not over
		if self.isAtLeastOneEmptyCell(matrix):
			return False

		#if no empty cells then we slide in all the directions, then we check if the score changed
		testMatrix = np.copy(matrix)
		score = 0

		testMatrix, score = self.slideMatrix(testMatrix, 'right', score)
		testMatrix, score = self.slideMatrix(testMatrix, 'up', score)
		testMatrix, score = self.slideMatrix(testMatrix, 'left', score)
		testMatrix, score = self.slideMatrix(testMatrix, 'down', score)

		return score == 0

	def test(self):
		"""
		testing stuff
		"""
		score = 0
		matrix = np.array([[0, 2, 2, 4], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
		self.printMatrix(matrix)
		print(score)

		matrix, score = self.slideMatrix(matrix, 'right', score)
		self.printUI(matrix, score)

		matrix, score = self.slideMatrix(matrix, 'up', score)
		self.printUI(matrix, score)

		matrix, score = self.slideMatrix(matrix, 'left', score)
		self.printUI(matrix, score)

		matrix, score = self.slideMatrix(matrix, 'down', score)
		self.printUI(matrix, score)

if __name__ == "__main__":
	game = Game2048AI()

	game.play()
