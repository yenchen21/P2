# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
import sys
from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        # Distance from nearest ghost
        nearest_ghost = manhattanDistance(newGhostStates[0].getPosition(), newPos) if newScaredTimes[0] == 0 else sys.maxint
        for i in range(1,len(newGhostStates)):
          curr_distance = manhattanDistance(newPos, newGhostStates[i].getPosition())
          if curr_distance < nearest_ghost and newScaredTimes[i] == 0:
            nearest_ghost = curr_distance 
        if nearest_ghost > 5:
          nearest_ghost = 5
        # Distance to nearest food?
        nearest_food = 0
        if newFood.count() > 0:
          nearest_food = manhattanDistance(newFood.asList()[0], newPos)
          for food in newFood.asList():
            curr_distance = manhattanDistance(food, newPos)
            if curr_distance < nearest_food:
              nearest_food = curr_distance
        food_count = newFood.count()
        if food_count == 0:
          food_count = 1
        if nearest_food == 0:
          nearest_food = .05
         
        return successorGameState.getScore() + 1/food_count + .05/nearest_food + 11*nearest_ghost

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

            gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """

        def value(gameState, depth, agentIndex):
          if agentIndex == gameState.getNumAgents():
            agentIndex = 0
            depth -= 1

          terminalState = gameState.isWin() or gameState.isLose()
          if depth == 0 or terminalState:
              return [self.evaluationFunction(gameState), Directions.STOP]
          
          if agentIndex == 0:
            return maxValue(gameState, depth)
          else:
            return minValue(gameState, depth, agentIndex)

        def maxValue(gameState, depth):
          legalMoves = gameState.getLegalActions(0)

          v = (-sys.maxint - 1)
          m = Directions.STOP
          for move in legalMoves:
            successorState = gameState.generateSuccessor(0, move)
            t = v
            v = max(v, value(successorState, depth, 1)[0])
            if t != v:
              m = move
          return [v, m]

        def minValue(gameState, depth, agentIndex):
          legalMoves = gameState.getLegalActions(agentIndex)

          v = sys.maxint
          m = Directions.STOP
          for move in legalMoves:
            successorState = gameState.generateSuccessor(agentIndex, move)
            t = v
            v = min(v, value(successorState, depth, agentIndex+1)[0])
            if t != v:
              m = move
          return [v, m]

        return value(gameState, self.depth, 0)[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """

        def value(gameState, depth, agentIndex, alpha, beta):
          if agentIndex == gameState.getNumAgents():
            agentIndex = 0
            depth -= 1

          terminalState = gameState.isWin() or gameState.isLose()
          if depth == 0 or terminalState:
              return [self.evaluationFunction(gameState), Directions.STOP]
          
          if agentIndex == 0:
            return maxValue(gameState, depth, alpha, beta)
          else:
            return minValue(gameState, depth, agentIndex, alpha, beta)

        def maxValue(gameState, depth, alpha, beta):
          legalMoves = gameState.getLegalActions(0)

          v = (-sys.maxint - 1)
          m = Directions.STOP
          for move in legalMoves:
            successorState = gameState.generateSuccessor(0, move)
            t = v
            v = max(v, value(successorState, depth, 1, alpha, beta)[0])
            if t != v:
              m = move
            if v > beta:
              return [v, m]
            alpha = max(alpha, v)
          return [v, m]

        def minValue(gameState, depth, agentIndex, alpha, beta):
          legalMoves = gameState.getLegalActions(agentIndex)

          v = sys.maxint
          m = Directions.STOP
          for move in legalMoves:
            successorState = gameState.generateSuccessor(agentIndex, move)
            t = v
            v = min(v, value(successorState, depth, agentIndex+1, alpha, beta)[0])
            if t != v:
              m = move
            if v < alpha:
              return [v, m]
            beta = min(beta, v)
          return [v, m]

        return value(gameState, self.depth, 0, (-sys.maxint - 1), sys.maxint)[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

