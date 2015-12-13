# myTeam.py
# ---------
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


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from approximateAdversarial import ApproximateAdversarialAgent

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'CautiousAttackAgent', second = 'HunterDefenseAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

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
      return -self.distancer.getDistance(
               myPosition, gameState.getInitialAgentPosition(self.index))
    else:
      return 2 * self.getScore(gameState) \
             - 100 * len(targetFood) \
             - min(map(lambda f: self.distancer.getDistance(myPosition, f),
                   targetFood))

class HunterDefenseAgent(ApproximateAdversarialAgent):
  """
  A defense-oriented agent that actively seeks out an enemy agent in its territory
  and tries to hunt it down
  """
  def evaluateState(self, gameState):
    myPosition = gameState.getAgentState(self.index).getPosition()
    targetFood = self.getFood(gameState).asList()

    score = 0

    for opponent in ApproximateAdversarialAgent.getOpponents(self, gameState):
      if abs(myPosition[0] - gameState.getInitialAgentPosition(opponent)[0]) < \
         abs(myPosition[0] - gameState.getInitialAgentPosition(self.index)[0]):
        return -1000000

      if not self.agentIsPacman(opponent, gameState):
        score += 1000

      score -= self.distancer.getDistance(myPosition, gameState.getAgentState(opponent).getPosition())

    return score


