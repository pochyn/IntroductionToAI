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

        "*** YOUR CODE HERE ***"
        # find distance to closest food
        if len(newFood.asList()) > 0:
            closest_food = 99999
        else:
            closest_food = -99999
        for food in newFood.asList():
            if getDistance(food, newPos) < closest_food:
                closest_food = getDistance(food, newPos)

        # find position of closest Ghost and distance to it
        if len(newGhostStates) > 0:
            closest_ghost = 9999999
        else:
            closest_ghost = -99999
        for ghost in newGhostStates:
            if getDistance(newPos, ghost.getPosition()) < closest_ghost:
                closest_ghost = getDistance(newPos, ghost.getPosition())

        # if power pallet eaten - chase and eat the ghost
        if newScaredTimes[0] != 0:
            return 500 * successorGameState.getScore() - 500*closest_ghost

        #do not approach ghost to close
        if closest_ghost < 2:
            return -9999999
        # main evaluation by quantity of food left, if eqaul - choose the closest food

        return - 100 * len(newFood.asList()) - closest_food + closest_ghost


def getDistance (point1, point2):
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])


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
        return self.DFMiniMax(gameState, 0, 0)[0]

    def DFMiniMax(self, pos, currDepth, currIndx):
        best_move = None

        #check terminal nodes
        if pos.isWin() or pos.isLose() or currDepth >= self.depth:
            return best_move, self.evaluationFunction(pos)

        #set up values for min and max
        if currIndx == 0:
            value = -9999999
        else:
            value = 9999999

        # get all moves
        actions = pos.getLegalActions(currIndx)
        for move in actions:
            #check if next agent is pacman if yes, the new depth
            nxt_pos = pos.generateSuccessor(currIndx, move)
            nxt_indx = (currIndx + 1) % pos.getNumAgents()
            nxt_depth = currDepth
            if nxt_indx == 0:
                nxt_depth += 1
            #recursive call
            nxt_move, nxt_val = self.DFMiniMax(nxt_pos, nxt_depth, nxt_indx)
            #new minimax values and moves
            if currIndx == 0 and value < nxt_val:
                value, best_move = nxt_val, move
            if currIndx != 0 and value > nxt_val:
                value, best_move = nxt_val, move
        return best_move, value


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.AlphaBeta(gameState, 0, 0,  -99999, 99999)[0]

    def AlphaBeta(self, pos, currDepth, currIndx, alpha, beta):
        best_move = None

        # check terminal nodes
        if pos.isWin() or pos.isLose() or currDepth >= self.depth:
            return best_move, self.evaluationFunction(pos)

        # set up values for min and max
        if currIndx == 0:
            value = -9999999
        else:
            value = 9999999
        # get all moves
        actions = pos.getLegalActions(currIndx)
        for move in actions:
            # check if next agent is pacman if yes, the new depth
            nxt_pos = pos.generateSuccessor(currIndx, move)
            nxt_indx = (currIndx + 1) % pos.getNumAgents()
            nxt_depth = currDepth
            if nxt_indx == 0:
                nxt_depth += 1
            # recursive call
            nxt_move, nxt_val = self.AlphaBeta(nxt_pos, nxt_depth, nxt_indx, alpha, beta)
            # new values, and possible pruning
            if currIndx == 0 and value < nxt_val:
                if value < nxt_val:
                    value, best_move = nxt_val, move
                if value >= beta:
                    return best_move, value
                alpha = max(alpha, value)
            if currIndx != 0 and value > nxt_val:
                if value > nxt_val:
                    value, best_move = nxt_val, move
                if value <= alpha:
                    return best_move, value
                beta = min(beta, value)

        return best_move, value

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
        return self.Expectimax(gameState, 0, 0)[0]

    def Expectimax(self, pos, currDepth, currIndx):
        best_move = None

        # check terminal nodes
        if pos.isWin() or pos.isLose() or currDepth >= self.depth:
            return best_move, self.evaluationFunction(pos)

        # set up values for min and max
        if currIndx == 0:
            value = -9999999
        else:
            value = 0
        # get all moves
        actions = pos.getLegalActions(currIndx)
        for move in actions:
            # check if next agent is pacman if yes, the new depth
            nxt_pos = pos.generateSuccessor(currIndx, move)
            nxt_indx = (currIndx + 1) % pos.getNumAgents()
            nxt_depth = currDepth
            if nxt_indx == 0:
                nxt_depth += 1
            # recursive call
            nxt_move, nxt_val = self.Expectimax(nxt_pos, nxt_depth, nxt_indx)
            # new values, no best_move for non pacman (CHANCE player)
            if currIndx == 0:
                if nxt_val > value:
                    value = nxt_val
                    best_move = move
            if currIndx != 0:
                value += 1.0 / len(actions) * nxt_val

        return best_move, value


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>

    1. if pacman see capsule he will it eat
    2. len(food) basically main evaluation, trying to get rid of all food
    3. getScore() needed basically only when pacman see posibility to eat ghost, in other cases always the same
    4. eat closest food
    5. try to be near closer capsule
    6. stay further from ghost (less important, pacman reacts fast)

    -- importance in descending order
    -- using negative values not reciprocal
    -- not sure but may be the case when ghost is far, pacman ate all food in his depth, so should wait until ghost come
    """
    "*** YOUR CODE HERE ***"


    #closest capsule to pacman
    closest_capsul = 999999
    if len(currentGameState.getCapsules()):
        for capsule in currentGameState.getCapsules():
            distance = getDistance(capsule, currentGameState.getPacmanPosition())
            if distance < closest_capsul:
                closest_capsul = distance
    else:
        closest_capsul = -9999999999

    # closest ghost to pacman
    closest_ghost = 99999
    if len(currentGameState.getGhostStates()) > 0:
        for ghost in currentGameState.getGhostStates():
            distance = getDistance(ghost.getPosition(), currentGameState.getPacmanPosition())
            if distance < closest_ghost:
                closest_ghost = (distance, ghost)
    else:
        closest_ghost = (-9999999999, None)

    # closest food to pacman
    closest_food = 999999
    if len(currentGameState.getFood().asList()) > 0:
        for food in currentGameState.getFood().asList():
            distance = getDistance(food, currentGameState.getPacmanPosition())
            if distance < closest_food:
                closest_food = distance
    else:
        closest_food = -9999999999

    return - 500*len(currentGameState.getCapsules()) - 100*len(currentGameState.getFood().asList()) + currentGameState.getScore() - 10*closest_food - 5*closest_capsul - closest_ghost[0]

# Abbreviation
better = betterEvaluationFunction
