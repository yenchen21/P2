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
        currentGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = currentGameState.getPacmanPosition()
        newFood = currentGameState.getFood()
        newGhostStates = currentGameState.getGhostStates()
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
         
        return currentGameState.getScore() + 10/food_count + .05/nearest_food + 11*nearest_ghost

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
            return expValue(gameState, depth, agentIndex)

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

        def expValue(gameState, depth, agentIndex):
          legalMoves = gameState.getLegalActions(agentIndex)

          v = 0.0
          for move in legalMoves:
            successorState = gameState.generateSuccessor(agentIndex, move)
            probability = 1.0/len(legalMoves)
            v += probability * value(successorState, depth, agentIndex+1)[0]
          return [v, None]

        return value(gameState, self.depth, 0)[1]

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    # Useful information you can extract from a GameState (pacman.py)
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

   
    def calcGhostScore(ghostStates, scaredTimes, pos):
        nearest_ghost = manhattanDistance(ghostStates[0].getPosition(), pos) if scaredTimes[0] == 0 else sys.maxint
        nearest_scared = False
        for i in range(1,len(ghostStates)):
          curr_distance = manhattanDistance(pos, ghostStates[i].getPosition())
          if curr_distance < nearest_ghost:
            nearest_scared = scaredTimes[i] != 0
            nearest_ghost = curr_distance 
        if nearest_scared:
          nearest_ghost = 5*nearest_ghost
        else:
          if nearest_ghost > 5:
            nearest_ghost = 5
          else:
            # If ghost is close, look for nearest capsule
            if nearest_ghost == 0:
              nearest_ghost = 1
            nearest_ghost = 1.0/(nearest_ghost*11)
        return nearest_ghost

    def calcFoodScore(foodPos, pos):
       # Distance to nearest food?
      furthest_food = 0
      nearest_food = 0
      if foodPos.count() > 0:
        furthest_food = manhattanDistance(foodPos.asList()[0], pos)
        nearest_food = furthest_food
        for food in foodPos.asList():
          curr_distance = manhattanDistance(food, pos)
          if curr_distance > furthest_food:
            furthest_food = curr_distance
          if curr_distance < nearest_food:
            nearest_food = curr_distance
      food_count = foodPos.count()
      if food_count == 0:
        food_count = 1
      if furthest_food == 0:
        furthest_food = .05
      if nearest_food == 0:
        nearest_food = .05
      return (10/food_count, .05/nearest_food, .05/furthest_food)


    def calcCapsuleScore(capsules, nearestGhost, pos): 
        nearest_capsule = sys.maxint 
        if len(capsules) > 0:
          nearest_capsule = manhattanDistance(capsules[0], pos)
          for capsule in capsules:
            curr_distance = manhattanDistance(capsule, pos)
            if curr_distance < nearest_capsule:
              nearest_capsule = curr_distance
          if nearest_capsule > 5 or nearestGhost > 5:
            nearest_capsule = sys.maxint
        return 1.0/nearest_capsule
    foodCount, nearestFood, furthestFood = calcFoodScore(newFood, newPos) 
    nearestGhost = calcGhostScore(newGhostStates, newScaredTimes, newPos)
    nearestCapsule = calcCapsuleScore(currentGameState.getCapsules(), nearestGhost, newPos)
    
    return currentGameState.getScore() + foodCount + nearestFood + furthestFood + nearestGhost

# Abbreviation
better = betterEvaluationFunction

