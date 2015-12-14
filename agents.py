from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game

import pdb

SEARCH_DEPTH = 7

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
                      conf, self.agentIsPacman(opponent, gameState, True))

    # Run negamax alpha-beta search to pick an optimal move
    bestVal, bestAction = float("-inf"), None
    for opponent in self.getOpponents(gameState):
      value, action = self.negamax(opponent,
                                   probableState,
                                   SEARCH_DEPTH,
                                   float("-inf"),
                                   float("inf"),
                                   1,
                                   retAction=True)
      if value > bestVal:
        bestVal, bestAction = value, action

    return action

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

    if not updatedBeliefs.totalCount():
      self.initializeBeliefs(agent)
    else:
      updatedBeliefs.normalize()
      self.positionBeliefs[agent] = updatedBeliefs

  def guessPosition(self, agent):
    """
    Return the most likely position of the given agent in the game.
    """
    return self.positionBeliefs[agent].argMax()

  def negamax(self, opponent, state, depth, alpha, beta, sign, retAction=False):
    """
    Negamax variation of minimax, with alpha-beta pruning.
    """
    if sign == 1:
      agent = self.index
    else:
      agent = opponent

    actions = state.getLegalActions(agent)
    actions.remove(Directions.STOP)
    actions.append(Directions.STOP)

    bestAction = None
    if not (depth and actions):
      bestVal = sign * self.evaluateState(state)
    else:
      bestVal = float("-inf")
      for action in actions:
        successor = state.generateSuccessor(agent, action)
        value = -self.negamax(opponent, successor, depth - 1, -alpha, -beta, -sign)
        if value > bestVal:
          bestVal, bestAction = value, action
        alpha = max(alpha, value)
        if alpha >= beta:
          break
    if retAction:
      return bestVal, bestAction
    else:
      return bestVal

  def agentIsPacman(self, agent, gameState, guessPosition=False):
    if guessPosition:
      agentPos = self.guessPosition(agent)
    else:
      agentPos = gameState.getAgentState(agent).getPosition()
    return (gameState.isRed(agentPos) != gameState.isOnRedTeam(agent))

  def evaluateState(self, gameState):
    """
    Evaluate the utility of a game state.
    """
    util.raiseNotDefined()

class CautiousAttackAgent(ApproximateAdversarialAgent):
  """
  An attack-oriented agent that will retreat back to its home zone
  after consuming 5 pellets.
  """
  def registerInitialState(self, gameState):
    ApproximateAdversarialAgent.registerInitialState(self, gameState)
    self.retreating = False

  def chooseAction(self, gameState):
    if (gameState.getAgentState(self.index).numCarrying < 5 and
        len(self.getFood(gameState).asList())):
      self.retreating = False
    else:
      self.retreating = True

    return ApproximateAdversarialAgent.chooseAction(self, gameState)

  def evaluateState(self, gameState):
    myPosition = gameState.getAgentState(self.index).getPosition()
    targetFood = self.getFood(gameState).asList()

    if self.retreating:
      return  -len(targetFood) \
              -self.distancer.getDistance(
               myPosition, gameState.getInitialAgentPosition(self.index))
    else:
      foodDistances = [self.distancer.getDistance(myPosition, food)
                       for food in targetFood]
      minDistance = min(foodDistances) if len(foodDistances) else 0
      return 2 * self.getScore(gameState) \
             -100 * len(targetFood) \
             -minDistance


class OpportunisticAttackAgent(ApproximateAdversarialAgent):
  def evaluateState(self, gameState):
    myPosition = gameState.getAgentState(self.index).getPosition()
    food = self.getFood(gameState).asList()

    targetFood = None
    maxDist = 0

    if not food or gameState.getAgentState(self.index).numCarrying > self.getScore(gameState):
      return 20*self.getScore(gameState) - self.distancer.getDistance(myPosition, gameState.getInitialAgentPosition(self.index))

    for f in food:
      d = min(map(lambda o: self.distancer.getDistance(gameState.getAgentState(o).getPosition(), f), self.getOpponents(gameState)))
      if d > maxDist:
        targetFood = f
        maxDist = d

    return 2 * self.getScore(gameState) \
           -100 * len(food) \
           -self.distancer.getDistance(myPosition, targetFood)


class GoalieAgent(ApproximateAdversarialAgent):
  """
  A defense-oriented agent that tries to place itself between its team's
  food and the closest opponent.
  """
  def evaluateState(self, gameState):
    myPosition = gameState.getAgentState(self.index).getPosition()
    shieldedFood = self.getFoodYouAreDefending(gameState).asList()
    opponentPositions = [gameState.getAgentState(opponent).getPosition()
                         for opponent in self.getOpponents(gameState)]

    if len(shieldedFood):
      opponentDistances = [(f, o, self.distancer.getDistance(f, o))
                           for f in shieldedFood for o in opponentPositions]
      atRiskFood, threateningOpponent = min(opponentDistances, key=lambda t: t[2])[0:2]
      return 1000 \
             -2 * self.distancer.getDistance(myPosition, atRiskFood) \
             -self.distancer.getDistance(myPosition, threateningOpponent)
    else:
      return -min([self.distancer.getDistance(myPosition, o) for o in opponentPositions])

class HunterDefenseAgent(ApproximateAdversarialAgent):
  """
  A defense-oriented agent that actively seeks out an enemy agent in its territory
  and tries to hunt it down
  """
  def evaluateState(self, gameState):
    myPosition = gameState.getAgentState(self.index).getPosition()
    if self.agentIsPacman(self.index, gameState):
        return -1000000

    score = 0
    for opponent in ApproximateAdversarialAgent.getOpponents(self, gameState):
      if not self.agentIsPacman(opponent, gameState):
        score += 1000
      else:
        score -= self.distancer.getDistance(
          myPosition, gameState.getAgentState(opponent).getPosition())

    return score
