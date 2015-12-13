from captureAgents import CaptureAgent
import util
from game import Directions
import game

SEARCH_DEPTH = 2

class ApproximateAdversarialAgent(CaptureAgent):
  """
  Superclass for agents choosing actions via alpha-beta search, with
  positions of unseen enemies approximated by Bayesian inference
  """
  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)

    # Get all non-wall positions on the board
    self.legalPositions = gameState.data.layout.walls.asList(False)

    # Initialize position belief distributions for opponents
    self.positionBeliefs = {}
    for opponent in self.getOpponents(gameState):
      self.initializeBeliefs(opponent)

  def initializeBeliefs(self, agent):
    """
    Uniformly initialize belief distributions for opponent positions.
    """
    self.positionBeliefs[agent] = util.Counter()
    for p in self.legalPositions:
      self.positionBeliefs[agent][p] = 1.0

  def chooseAction(self, gameState):
    # Update belief distribution about opponent positions and place hidden
    # opponents in their most likely positions
    myPosition = gameState.getAgentState(self.index).getPosition()
    noisyDistances = gameState.getAgentDistances()
    probableState = gameState.deepCopy()

    for opponent in self.getOpponents(gameState):
      pos = gameState.getAgentPosition(opponent)
      if pos:
        self.fixPosition(opponent, pos)
      else:
        self.elapseTime(opponent, gameState)
        self.observe(opponent, noisyDistances[opponent], gameState)

      self.displayDistributionsOverPositions(self.positionBeliefs.values())
      probablePosition = self.guessPosition(opponent)
      conf = game.Configuration(probablePosition, Directions.STOP)
      probableState.data.agentStates[opponent] = game.AgentState(
                      conf, self.opponentInTerritory(gameState, opponent))

    # Run alpha-beta search to pick an optimal move
    return self.alphabeta(probableState, self.index,
                          SEARCH_DEPTH,
                          float("-inf"), float("inf"),
                          retAction=True)

  def fixPosition(self, agent, position):
    """
    Fix the position of an opponent in an agent's belief distributions.
    """
    updatedBeliefs = util.Counter()
    updatedBeliefs[position] = 1.0
    self.positionBeliefs[agent] = updatedBeliefs

  def elapseTime(self, agent, gameState):
    """
    Elapse belief distributions for an agent's position by one time step.
    """
    updatedBeliefs = util.Counter()
    for (oldX, oldY), oldProbability in self.positionBeliefs[agent].items():
      newDist = util.Counter()
      for p in [(oldX - 1, oldY), (oldX + 1, oldY),
                (oldX, oldY - 1), (oldX, oldY + 1)]:
        if p in self.legalPositions:
          newDist[p] = 1.0
      newDist.normalize()
      for newPosition, newProbability in newDist.items():
        updatedBeliefs[newPosition] += newProbability * oldProbability
    self.positionBeliefs[agent] = updatedBeliefs

  def observe(self, agent, noisyDistance, gameState):
    """
    Update belief distributions for an agent's position based upon
    a noisy distance measurement for that agent.
    """
    myPosition = gameState.getAgentState(self.index).getPosition()
    updatedBeliefs = util.Counter()
    for p in self.legalPositions:
      trueDistance = util.manhattanDistance(myPosition, p)
      positionProbability = gameState.getDistanceProb(trueDistance, noisyDistance)
      updatedBeliefs[p] = positionProbability * self.positionBeliefs[agent][p]

    updatedBeliefs.normalize()
    self.positionBeliefs[agent] = updatedBeliefs

  def guessPosition(self, agent):
    """
    Return the most likely position of the given agent in the game.
    """
    return self.positionBeliefs[agent].argMax()

  def alphabeta(self, state, agent, depth, alpha, beta, retAction=False):
    """
    Alpha-beta pruning adaptation
    """
    actions = state.getLegalActions(agent)
    actions.remove(Directions.STOP)
    bestAction = None
    if not (depth and actions):
      bestVal = self.evaluateState(state)
    else:
      if agent in self.getTeam(state):
        bestVal = float("-inf")
      else:
        bestVal = float("inf")

      nextAgent = (agent + 1) % state.getNumAgents()
      nextDepth = depth - 1 if nextAgent == self.index else depth

      for action in actions:
        successor = state.generateSuccessor(agent, action)
        value = self.alphabeta(successor, nextAgent, nextDepth, alpha, beta)
        if state.isOnRedTeam(agent):
          if value > bestVal:
            bestVal, bestAction = value, action
          if bestVal > beta:
            break
          alpha = max(alpha, bestVal)
        else:
          if value < bestVal:
            bestVal, bestAction = value, action
          if bestVal < alpha:
            break
          beta = min(beta, bestVal)
    if retAction:
      return bestAction
    else:
      return bestVal

  def opponentInTerritory(self, gameState, oppIndex):
    return abs(self.guessPosition(oppIndex)[0] - gameState.getInitialAgentPosition(self.index)[0]) < \
           abs(self.guessPosition(oppIndex)[0] - gameState.getInitialAgentPosition(oppIndex)[0])

  def evaluateState(self, gameState):
    """
    Evaluate the utility of a game state.
    """
    util.raiseNotDefined()
