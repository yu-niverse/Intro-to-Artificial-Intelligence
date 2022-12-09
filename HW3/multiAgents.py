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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        minGhostDistance = min([manhattanDistance(newPos, state.getPosition()) for state in newGhostStates])

        scoreDiff = childGameState.getScore() - currentGameState.getScore()

        pos = currentGameState.getPacmanPosition()
        nearestFoodDistance = min([manhattanDistance(pos, food) for food in currentGameState.getFood().asList()])
        newFoodsDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        newNearestFoodDistance = 0 if not newFoodsDistances else min(newFoodsDistances)
        isFoodNearer = nearestFoodDistance - newNearestFoodDistance

        direction = currentGameState.getPacmanState().getDirection()
        if minGhostDistance <= 1 or action == Directions.STOP:
            return 0
        if scoreDiff > 0:
            return 8
        elif isFoodNearer > 0:
            return 4
        elif action == direction:
            return 2
        else:
            return 1





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
    Your minimax agent (Part 1)
    """
    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        # Begin your code (Part 1)

        # the recursion function implementing the minimax agent
        def implementMinimax(depth, agent, state):

            # return the evaluation score when Pacman comes to the end state or overceeding the depth limit
            if state.isWin() or state.isLose() or depth > self.depth:
                return self.evaluationFunction(state)

            # getting all legal actions
            actions = state.getLegalActions(agent)
            # storing the score of every possible action in a list
            actionScores = []
            for action in actions:
                nextState = state.getNextState(agent, action)
                # all agents have moved -> back to Pacman's turn and start another round
                if (agent + 1) == state.getNumAgents(): 
                    actionScores.append(implementMinimax(depth + 1, 0, nextState))
                # agent taking turns
                else: actionScores.append(implementMinimax(depth, agent + 1, nextState))

            # performing the minimax procedure
            # 1. Pacman : return the maximum action score
            if agent == 0: 
                if depth == 1: # return the next action when it comes back to the root
                    for i in range(len(actionScores)):
                        if actionScores[i] == max(actionScores): return actions[i]
                else: actionScore = max(actionScores)
            # 2. Ghosts : return the minimum action score
            else: actionScore = min(actionScores)
            return actionScore

        # implement minimax agent
        return implementMinimax (1, 0, gameState)
        # End your code (Part 1)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (Part 2)
    """
    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        # Begin your code (Part 2)
        def implementAlphaBeta(depth, agent, state, alpha, beta):

            # return the evaluation score when Pacman comes to the end state or overceeding the depth limit
            if (state.isWin() or state.isLose() or depth > self.depth):
                return self.evaluationFunction(state)

            # getting all legal actions
            actions = state.getLegalActions(agent)
            # storing the score of every possible action in a list
            actionScores = []
            for action in actions:
                nextState = state.getNextState(agent, action)
                # all agents have moved -> back to Pacman's turn and start another round
                if (agent + 1) == state.getNumAgents():
                    actionScore = implementAlphaBeta(depth + 1, 0, nextState, alpha, beta)
                    actionScores.append(actionScore)
                # agent taking turns
                else: 
                    actionScore = implementAlphaBeta(depth, agent + 1, nextState, alpha, beta)
                    actionScores.append(actionScore)

                # pruning the branches
                # 1. for Pacman / the maximizer
                if agent == 0:
                    if actionScore > beta: return actionScore
                    alpha = max(alpha, actionScore)
                # 2. for the ghosts / the minimizer
                else:
                    if actionScore < alpha: return actionScore
                    beta = min(beta, actionScore)

            # performing the minimax procedure
            # 1. Pacman : return the maximum action score
            if agent == 0: 
                if depth == 1: # return the next action when it comes back to the root
                    for i in range(len(actionScores)):
                        if actionScores[i] == max(actionScores): return actions[i]
                else: actionScore = max(actionScores)
            # 2. Ghosts : return the minimum action score
            else: actionScore = min(actionScores)
            return actionScore

        # initialize alpha & beta
        alpha = -99999
        beta = 99999
        # implement alpha-beta pruning
        return implementAlphaBeta(1, 0, gameState, alpha, beta)
        # End your code (Part 2)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (Part 3)
    """
    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        # Begin your code (Part 3)
        def implementExpectimax(depth, agent, state):

            # return the evaluation score when Pacman comes to the end state or overceeding the depth limit
            if (state.isWin() or state.isLose() or depth > self.depth):
                return self.evaluationFunction(state)

            # getting all legal actions
            actions = state.getLegalActions(agent)
            # storing the score of every possible action in a list
            actionScores = []
            for action in actions:
                nextState = state.getNextState(agent, action)
                # all agents have moved -> back to Pacman's turn and start another round
                if (agent + 1) == state.getNumAgents():
                    actionScores.append(implementExpectimax(depth + 1, 0, nextState))
                # agent taking turns
                else: actionScores.append(implementExpectimax(depth, agent + 1, nextState))


            # performing the expectimax procedure
            # 1. Pacman : return the maximum action score
            if agent == 0: 
                if depth == 1: # return the next action when it comes back to the root
                    for i in range(len(actionScores)):
                        if actionScores[i] == max(actionScores): return actions[i]
                else: actionScore = max(actionScores)
            # 2. Ghosts : choose legal action uniformly at random
            #    -> now returning the average case instead of the minimum
            else: actionScore = float(sum(actionScores) / len(actionScores))
            return actionScore

        # implement the expectimax agent
        return implementExpectimax(1, 0, gameState)
        # End your code (Part 3)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (Part 4).
    """
    # Begin your code (Part 4)

    # Accessing useful information for my evaluation function :
    score = currentGameState.getScore() # current score
    position = currentGameState.getPacmanPosition() # Pacman's current position
    food = currentGameState.getFood() # list of foods
    capsules = currentGameState.getCapsules() # list of capsules
    ghostStates = currentGameState.getGhostStates() # ghosts' states

    # Giving different weights to some particular states
    # -> to control Pacman's action / strategy
    WEIGHT_FOOD = 10.0
    WEIGHT_CAPSULE = 25.0
    WEIGHT_GHOST = -10.0 
    WEIGHT_SCARED_GHOST = 300.0 

    # set a higher score while approaching capsules
    capsuleDistances = [manhattanDistance(position, capsulePosition) for capsulePosition in capsules]
    if len(capsuleDistances): score += WEIGHT_CAPSULE / min(capsuleDistances)
    else: score += WEIGHT_FOOD

    # set a higher score while approaching food
    foodDistances = [manhattanDistance(position, foodPosition) for foodPosition in food.asList()]
    if len(foodDistances): score += WEIGHT_FOOD / min(foodDistances) 
    else: score += WEIGHT_FOOD

    # interactions with the ghosts
    for ghost in ghostStates:
        distance = manhattanDistance(position, ghost.getPosition())
        if distance > 0:
            # set a higher score if approaching scared ghosts
            if ghost.scaredTimer > 0: score += WEIGHT_SCARED_GHOST / distance
            # lower the score when the ghost is close
            else: score += WEIGHT_GHOST / distance
    
    return score
    # End your code (Part 4)

# Abbreviation
better = betterEvaluationFunction
