from captureAgents import CaptureAgent
from capture import SIGHT_RANGE
import random, time, util
from game import Directions
import game

class ApproximateAdversarialAgent(CaptureAgent):
  """
  Superclass for agents choosing actions via alpha-beta search, with
  positions of unseen enemies approximated by Bayesian inference
  """
  #####################
  # AI algorithm code #
  #####################

  SEARCH_DEPTH = 5

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
    for opponent in self.getOpponents(gameState):
      probablePosition = self.guessPosition(opponent)
      conf = game.Configuration(probablePosition, Directions.STOP)
      probableState.data.agentStates[opponent] = game.AgentState(
        conf, probableState.isRed(probablePosition) != probableState.isOnRedTeam(opponent))

    # Run negamax alpha-beta search to pick an optimal move
    bestVal, bestAction = float("-inf"), None
    for opponent in self.getOpponents(gameState):
      value, action = self.expectinegamax(opponent,
                                          probableState,
                                          self.SEARCH_DEPTH,
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
    Assume opponents move randomly, but also check for any food lost from
    the previous turn.
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

    lastObserved = self.getPreviousObservation()
    if lastObserved:
      lostFood = [food for food in self.getFoodYouAreDefending(lastObserved).asList()
                  if food not in self.getFoodYouAreDefending(gameState).asList()]
      for f in lostFood:
        updatedBeliefs[f] = 1.0/len(self.getOpponents(gameState))

    self.positionBeliefs[agent] = updatedBeliefs


  def observe(self, agent, noisyDistance, gameState):
    """
    Update belief distributions for an agent's position based upon
    a noisy distance measurement for that agent.
    """
    myPosition = self.getAgentPosition(self.index, gameState)
    teammatePositions = [self.getAgentPosition(teammate, gameState)
                         for teammate in self.getTeam(gameState)]
    updatedBeliefs = util.Counter()

    for p in self.legalPositions:
      if any([util.manhattanDistance(teammatePos, p) <= SIGHT_RANGE
              for teammatePos in teammatePositions]):
        updatedBeliefs[p] = 0.0
      else:
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

  def expectinegamax(self, opponent, state, depth, sign, retAction=False):
    """
    Negamax variation of expectimax.
    """
    if sign == 1:
      agent = self.index
    else:
      agent = opponent

    bestAction = None
    if self.stateIsTerminal(agent, state) or depth == 0:
      bestVal = sign * self.evaluateState(state)
    else:
      actions = state.getLegalActions(agent)
      actions.remove(Directions.STOP)
      bestVal = float("-inf") if agent == self.index else 0
      for action in actions:
        successor = state.generateSuccessor(agent, action)
        value = -self.expectinegamax(opponent, successor, depth - 1, -sign)
        if agent == self.index and value > bestVal:
          bestVal, bestAction = value, action
        elif agent == opponent:
          bestVal += value/len(actions)

    if agent == self.index and retAction:
      return bestVal, bestAction
    else:
      return bestVal

  def stateIsTerminal(self, agent, gameState):
    """
    Check if the search tree should stop expanding at the given game state
    on the given agent's turn.
    """
    return len(gameState.getLegalActions(agent)) == 0

  def evaluateState(self, gameState):
    """
    Evaluate the utility of a game state.
    """
    util.raiseNotDefined()

  #####################
  # Utility functions #
  #####################

  def getAgentPosition(self, agent, gameState):
    """
    Return the position of the given agent.
    """
    pos = gameState.getAgentPosition(agent)
    if pos:
      return pos
    else:
      return self.guessPosition(agent)

  def agentIsPacman(self, agent, gameState):
    """
    Check if the given agent is operating as a Pacman in its current position.
    """
    agentPos = self.getAgentPosition(agent, gameState)
    return (gameState.isRed(agentPos) != gameState.isOnRedTeam(agent))

  def getOpponentDistances(self, gameState):
    """
    Return the IDs of and distances to opponents, relative to this agent.
    """
    return [(o, self.distancer.getDistance(
             self.getAgentPosition(self.index, gameState),
             self.getAgentPosition(o, gameState)))
            for o in self.getOpponents(gameState)]

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
    myPosition = self.getAgentPosition(self.index, gameState)
    targetFood = self.getFood(gameState).asList()
    distanceFromStart = abs(myPosition[0] - gameState.getInitialAgentPosition(self.index)[0])
    opponentDistances = self.getOpponentDistances(gameState)
    opponentDistance = min([dist for id, dist in opponentDistances])

    if self.retreating:
      return  - len(targetFood) \
              - 2 * distanceFromStart \
              + opponentDistance
    else:
      foodDistances = [self.distancer.getDistance(myPosition, food)
                       for food in targetFood]
      minDistance = min(foodDistances) if len(foodDistances) else 0
      return 2 * self.getScore(gameState) \
             - 100 * len(targetFood) \
             - 3 * minDistance \
             + 2 * distanceFromStart \
             + opponentDistance


class OpportunisticAttackAgent(ApproximateAdversarialAgent):
  def evaluateState(self, gameState):
    myPosition = self.getAgentPosition(self.index, gameState)
    food = self.getFood(gameState).asList()

    targetFood = None
    maxDist = 0

    opponentDistances = self.getOpponentDistances(gameState)
    opponentDistance = min([dist for id, dist in opponentDistances])

    if not food or gameState.getAgentState(self.index).numCarrying > self.getScore(gameState) > 0:
      return 20 * self.getScore(gameState) \
             - self.distancer.getDistance(myPosition, gameState.getInitialAgentPosition(self.index)) \
             + opponentDistance

    for f in food:
      d = min([self.distancer.getDistance(self.getAgentPosition(o, gameState), f)
              for o in self.getOpponents(gameState)])
      if d > maxDist:
        targetFood = f
        maxDist = d
    if targetFood:
      foodDist = self.distancer.getDistance(myPosition, targetFood)
    else:
      foodDist = 0

    distanceFromStart = abs(myPosition[0] - gameState.getInitialAgentPosition(self.index)[0])
    if not len(food):
      distanceFromStart *= -1

    return 2 * self.getScore(gameState) \
           - 100 * len(food) \
           - 2 * foodDist \
           + opponentDistance \
           + distanceFromStart


class DefensiveAgent(ApproximateAdversarialAgent):
  """
  A defense-oriented agent that should never cross into the opponent's territory.
  """
  TERMINAL_STATE_VALUE = -1000000

  def stateIsTerminal(self, agent, gameState):
    return self.agentIsPacman(self.index, gameState) or \
      ApproximateAdversarialAgent.stateIsTerminal(self, agent, gameState)

class GoalieAgent(DefensiveAgent):
  """
  A defense-oriented agent that tries to place itself between its team's
  food and the closest opponent.
  """
  def evaluateState(self, gameState):
    if self.agentIsPacman(self.index, gameState):
      return DefensiveAgent.TERMINAL_STATE_VALUE

    myPosition = self.getAgentPosition(self.index, gameState)
    shieldedFood = self.getFoodYouAreDefending(gameState).asList()
    opponentPositions = [self.getAgentPosition(opponent, gameState)
                         for opponent in self.getOpponents(gameState)]

    if len(shieldedFood):
      opponentDistances = util.Counter()
      opponentTotalDistances = util.Counter()

      for f in shieldedFood:
        for o in opponentPositions:
          distance = self.distancer.getDistance(f, o)
          opponentDistances[(f, o)] = distance
          opponentTotalDistances[o] -= distance

      threateningOpponent = opponentTotalDistances.argMax()
      atRiskFood, shortestDist = None, float("inf")
      for (food, opponent), dist in opponentDistances.iteritems():
        if opponent == threateningOpponent and dist < shortestDist:
          atRiskFood, shortestDist = food, dist

      return len(shieldedFood) \
             - 2 * self.distancer.getDistance(myPosition, atRiskFood) \
             - self.distancer.getDistance(myPosition, threateningOpponent)
    else:
      return -min(self.getOpponentDistances(gameState), key=lambda t: t[1])[1]

class HunterDefenseAgent(DefensiveAgent):
  """
  A defense-oriented agent that actively seeks out an enemy agent in its territory
  and tries to hunt it down
  """
  def evaluateState(self, gameState):
    myPosition = self.getAgentPosition(self.index, gameState)
    if self.agentIsPacman(self.index, gameState):
        return DefensiveAgent.TERMINAL_STATE_VALUE

    score = 0
    pacmanState = [self.agentIsPacman(opponent, gameState)
                   for opponent in self.getOpponents(gameState)]
    opponentDistances = self.getOpponentDistances(gameState)

    for isPacman, (id, distance) in zip(pacmanState, opponentDistances):
      if isPacman:
        score -= 100000
        score -= 5 * distance
      elif not any(pacmanState):
        score -= distance

    return score
