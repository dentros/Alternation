# This is the code of the experiment of MBoE .
# for agents with type-B state representation (position + last winners)


# Please run it on Piecharm or some relevant IDE, Not anaconda so multiprocessing works

number_of_max_agents = 10 #The script runs simulteanously the experiment for any number of agents (multiprocessing) 
number_of_possible_positions_per_agent = 3 # numbers of possible positions in the grid including the first and the final one
number_of_possible_actions_per_agent = 2 
number_of_possible_terminal_states_per_agent=2
numberofepisodes = 10000

import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import multiprocessing
import math


def encodeState(currentPositions, l, numberofagents):
# This function encodes states of multiple agents in a list for use in Q learning.


    a = int(''.join(str(x) for x in currentPositions), number_of_possible_positions_per_agent) #this line encodes the list of current agents' positions, with an integer in the base of number_of_possible_positions_per_agent. 
	#E.g for agent positions [0,0,2], a=2 . For agent positions [0,1,0=, a=3 etc  
	
    b = int(''.join(str(y) for y in l), number_of_possible_terminal_states_per_agent) #this line encodes the list of last episode's top agents, with an integer in the base of terminal_states_per_agent. 
	#E.g for last episode's top [0,0,1], b=1 . for last episode's top agents [0,1,0]=, b=2 etc  
    done = False
    d = pow(number_of_possible_positions_per_agent, numberofagents)
    c = 0
    while not done:
        d = d / 10
        c += 1
        if int(d) is 0:
            done = True

    # print(    c)
    return pow(10, c) * b + a  
    # Finally, the state is encoded as one unique integer. The digits at the left of the integer encode last episode and the rest digits at the right the current position
	#E.g 1: For last episode's top agents [1,1,1,1] and current positions [0,0,0,0] we get 1500. 
	#E.g 2: While, for last episode's top agents [0,0,0,0] and current positions [2,2,2,2] we get 80.
	#E.g 3: For last episode's top agents [1,1,1,1] and current positions [2,2,2,2] we get 1580


def alternation(numberofepisodes, numberofagents, terminalOccurencesPerEpisode, TopAgentsPerEpisode):
    ####### ALTERNATION METRICS !!!!!!!!!!!!!!
	####### In brief, this function gets the list of Top Agents FOR ALL THE EPISODES, returns the 6 ALT metrics and plots the 6 cumulative ALT metrics, both for the last 10% of episodes and all episodes. 
	
    numberOfBatches = numberofepisodes - (numberofagents - 1) #find the number of overlapping episode batches, these episodes can be divided into
    # print("Number of batches", numberOfBatches)

    termOccPerBatch = numberOfBatches * [0]
    numOfWinnersPerBatch = numberOfBatches * [0]

    betaFALT = numberOfBatches * [0]
    betaEALT = numberOfBatches * [0]
    betaEFALT = numberOfBatches * [0]
    betaEEALT = numberOfBatches * [0]
    betaCALT = numberOfBatches * [0]
    betaAALT = numberOfBatches * [0]

	#We run each batch to evaluate it:
    for batchId in range(0, numberOfBatches):
        #         print(    "BATCH: ", batchId)


		###################  Measure require quantities and lists required for each ALT version:
		
        whoReachedTopInThisBatch = numberofagents * [
            0]  # one element per episode of the batch. It calculates how many agents managed to reach the top at each episode of the batch
        exclusiveWinningEpisodesInThisBatch = 0  # needed for CALT
        numberOfWinningsInBatchPerAgent = numberofagents * [0]  # needed for AALT

        for eb in range(batchId, batchId + numberofagents):  # run within batch
            #             print(    "eb", eb)

            if sum(TopAgentsPerEpisode[eb]) is 1:  # if there is one exclusive winner in this episode
                exclusiveWinningEpisodesInThisBatch += 1  # needed for EALT (and EEALT)

            for i in range(numberofagents):  # for every agent
                if TopAgentsPerEpisode[eb][i] is 1:  # if this agent won in episode eb
                    whoReachedTopInThisBatch[i] = 1  # needed for CALT

                numberOfWinningsInBatchPerAgent[i] += TopAgentsPerEpisode[eb][i]

            # print(    "TopAgentsPerEpisode[eb]",TopAgentsPerEpisode[eb])
            numOfWinnersPerBatch[batchId] = sum(whoReachedTopInThisBatch)
            termOccPerBatch[batchId] += terminalOccurencesPerEpisode[eb]
		###############################################################################################
		
        ##########beta values for each batch (beta as defined in the thesis):

        betaFALT[batchId] = numOfWinnersPerBatch[batchId] / termOccPerBatch[batchId]
        betaEFALT[batchId] = pow(betaFALT[batchId], 2)
        betaEALT[batchId] = exclusiveWinningEpisodesInThisBatch * numOfWinnersPerBatch[batchId] / pow(numberofagents, 2)
        betaEEALT[batchId] = pow(betaEALT[batchId], 2)

        for eb in range(batchId, batchId + numberofagents):  # run within batch again :(
            betaCALT[batchId] += (numberofagents - sum(TopAgentsPerEpisode[eb])) * (betaEFALT[batchId])

        betaCALT[batchId] = betaCALT[batchId] / (numberofagents * (numberofagents - 1))

        countOfExclusiveWinnersInBatch = 0
        for i in range(numberofagents):
            if numberOfWinningsInBatchPerAgent[i] is 1:
                countOfExclusiveWinnersInBatch += 1

        betaAALT[batchId] = countOfExclusiveWinnersInBatch / termOccPerBatch[batchId]
        ####################################################################################################
		
    #  Cumulative beta lists: 
    cumBetaFALT = np.cumsum(betaFALT)
    cumBetaEALT = np.cumsum(betaEALT)
    cumBetaEFALT = np.cumsum(betaEFALT)
    cumBetaEEALT = np.cumsum(betaEEALT)
    cumBetaCALT = np.cumsum(betaCALT)
    cumBetaAALT = np.cumsum(betaAALT)

    #  Normalized Cumulative beta lists
    for i in range(len(cumBetaFALT)):
        cumBetaFALT[i] = cumBetaFALT[i] / (i + 1)
        cumBetaEALT[i] = cumBetaEALT[i] / (i + 1)
        cumBetaEFALT[i] = cumBetaEFALT[i] / (i + 1)
        cumBetaEEALT[i] = cumBetaEEALT[i] / (i + 1)
        cumBetaCALT[i] = cumBetaCALT[i] / (i + 1)
        cumBetaAALT[i] = cumBetaAALT[i] / (i + 1)


################### PRINTING THE ALL THE ALT VERSION VALUES + THE ESTIMATED EQUIVALENT OF PERFECT ALTERNATING AGENTS BASED ON THE BENCHMARK EVALUATION. 
################### e.g A result could be that the current behaviour of the system is like if 6.5 out of 10 agents alternated perfectly

    print("Final Total FALT for ", numberofagents, " agents : ", sum(betaFALT) / numberOfBatches,
          "Estimated Equivalent of Perfect Alternation among:", numberofagents * (sum(betaFALT) / numberOfBatches),
          "agents")
    print("Final Total EALT for ", numberofagents, " agents : ", sum(betaEALT) / numberOfBatches,
          "Estimated Equivalent of Perfect Alternation among:", numberofagents * (sum(betaEALT) / numberOfBatches),
          "agents")
    print("Final Total EFALT for ", numberofagents, " agents : ", sum(betaEFALT) / numberOfBatches,
          "Estimated Equivalent of Perfect Alternation among:",
          numberofagents * (((math.sqrt(sum(betaEFALT) / numberOfBatches)) - 0.000000000532183463) / 0.9999999999),
          "agents")
    print("Final Total EEALT for ", numberofagents, " agents : ", sum(betaEEALT) / numberOfBatches,
          "Estimated Equivalent of Perfect Alternation among:",
          numberofagents * (((math.sqrt(sum(betaEEALT) / numberOfBatches)) - 0.000000000532183463) / 0.9999999999),
          "agents")
    print("Final Total CALT for ", numberofagents, " agents : ", sum(betaCALT) / numberOfBatches,
          "Estimated Equivalent of Perfect Alternation among:",
          numberofagents * (math.sqrt(sum(betaCALT) / numberOfBatches) - 0.000000000674983614), "agents")
    if sum(betaAALT) / numberOfBatches is not 0:
        print("Final Total AALT for ", numberofagents, " agents : ", sum(betaAALT) / numberOfBatches,
              "Estimated Equivalent of Perfect Alternation among:",
              numberofagents * ((sum(betaAALT) / numberOfBatches)+1)/2,
              "agents")
    else:
        print("Final Total AALT for ", numberofagents, " agents : ", sum(betaAALT) / numberOfBatches,
              "No alternation is recognized")


################### Plotting! ###################
    plt.plot(cumBetaFALT, 'r-', cumBetaEALT, 'y-', cumBetaEFALT, 'o--', cumBetaEEALT, 'p--', cumBetaCALT, 'g-',
             cumBetaAALT, 'm-')
    plt.gca().legend(('Cummulative beta FALT', 'Cummulative beta EALT', 'Cummulative beta EFALT',
                      'Cummulative beta EEALT', 'Cummulative beta CALT', 'Cummulative beta AALT'))
    plt.xlabel('Batches')
    plt.ylim(top=1)
    plt.title('Number of agents: %i' % numberofagents)
    plt.show()

    #  Cumulative beta lists for the last 10% of batches
    lastTenPercentCumBetaFALT = np.cumsum(betaFALT[-(int(0.1 * numberOfBatches)):])
    lastTenPercentCumBetaEALT = np.cumsum(betaEALT[-(int(0.1 * numberOfBatches)):])
    lastTenPercentCumBetaEFALT = np.cumsum(betaEFALT[-(int(0.1 * numberOfBatches)):])
    lastTenPercentCumBetaEEALT = np.cumsum(betaEEALT[-(int(0.1 * numberOfBatches)):])
    lastTenPercentCumBetaCALT = np.cumsum(betaCALT[-(int(0.1 * numberOfBatches)):])
    lastTenPercentCumBetaAALT = np.cumsum(betaAALT[-(int(0.1 * numberOfBatches)):])

    # Normalized Cumulative beta lists for the last 10% of batches
    for i in range(len(lastTenPercentCumBetaFALT)):
        lastTenPercentCumBetaFALT[i] = lastTenPercentCumBetaFALT[i] / (i + 1)
        lastTenPercentCumBetaEALT[i] = lastTenPercentCumBetaEALT[i] / (i + 1)
        lastTenPercentCumBetaEFALT[i] = lastTenPercentCumBetaEFALT[i] / (i + 1)
        lastTenPercentCumBetaEEALT[i] = lastTenPercentCumBetaEEALT[i] / (i + 1)
        lastTenPercentCumBetaCALT[i] = lastTenPercentCumBetaCALT[i] / (i + 1)
        lastTenPercentCumBetaAALT[i] = lastTenPercentCumBetaAALT[i] / (i + 1)

################### Now, again for the last 10% of the episodes: 
################### PRINTING THE ALL THE ALT VERSION VALUES + THE ESTIMATED EQUIVALENT OF PERFECT ALTERNATING AGENTS BASED ON THE BENCHMARK EVALUATION. 
################### e.g A result could be that the current behaviour of the system is like if 6.5 out of 10 agents alternated perfectly

    print("Last 10% Batches' Final Total FALT for ", numberofagents, " agents : ",
          sum(betaFALT[-(int(0.1 * numberOfBatches)):]) / int(0.1 * numberOfBatches),
          "Estimated Equivalent of Perfect Alternation among:",
          numberofagents * sum(betaFALT[-(int(0.1 * numberOfBatches)):]) / int(0.1 * numberOfBatches), "agents")
    print("Last 10% Batches' Final Total EALT for ", numberofagents, " agents : ",
          sum(betaEALT[-(int(0.1 * numberOfBatches)):]) / int(0.1 * numberOfBatches),
          "Estimated Equivalent of Perfect Alternation among:",
          numberofagents * sum(betaEALT[-(int(0.1 * numberOfBatches)):]) / int(0.1 * numberOfBatches), "agents")
    print("Last 10% Batches' Final Total EFALT for ", numberofagents, " agents : ",
          sum(betaEFALT[-(int(0.1 * numberOfBatches)):]) / int(0.1 * numberOfBatches),
          "Estimated Equivalent of Perfect Alternation among:", numberofagents * (((math.sqrt(
            sum(betaEFALT[-(int(0.1 * numberOfBatches)):]) / int(
                0.1 * numberOfBatches))) - 0.000000000532183463) / 0.9999999999), "agents")
    print("Last 10% Batches' Final Total EEALT for ", numberofagents, " agents : ",
          sum(betaEEALT[-(int(0.1 * numberOfBatches)):]) / int(0.1 * numberOfBatches),
          "Estimated Equivalent of Perfect Alternation among:", numberofagents * (((math.sqrt(
            sum(betaEEALT[-(int(0.1 * numberOfBatches)):]) / int(
                0.1 * numberOfBatches))) - 0.000000000532183463) / 0.9999999999), "agents")
    print("Last 10% Batches' Final Total CALT for ", numberofagents, " agents : ",
          sum(betaCALT[-(int(0.1 * numberOfBatches)):]) / int(0.1 * numberOfBatches),
          "Estimated Equivalent of Perfect Alternation among:", numberofagents * (math.sqrt(
            sum(betaCALT[-(int(0.1 * numberOfBatches)):]) / int(0.1 * numberOfBatches)) - 0.000000000674983614),
          "agents")

    if sum(betaAALT[-(int(0.1 * numberOfBatches)):]) / int(0.1 * numberOfBatches) is not 0:
        print("Last 10% Batches' Final Total AALT for ", numberofagents, " agents : ",
              sum(betaAALT[-(int(0.1 * numberOfBatches)):]) / int(0.1 * numberOfBatches),
              "Estimated Equivalent of Perfect Alternation among:", numberofagents * (sum(
                betaAALT[-(int(0.1 * numberOfBatches)):]) +1)/2, "agents")
    else:
        print("Last 10% Batches' Final Total AALT for ", numberofagents, " agents : ",
              sum(betaAALT[-(int(0.1 * numberOfBatches)):]) / int(0.1 * numberOfBatches),
              "No alternation is recognized")

    plt.plot(lastTenPercentCumBetaFALT, 'r-', lastTenPercentCumBetaEALT, 'y-', lastTenPercentCumBetaEFALT, 'o--',
             lastTenPercentCumBetaEEALT, 'p--', lastTenPercentCumBetaCALT, 'g-',
             lastTenPercentCumBetaAALT, 'm-')
    plt.gca().legend(('Last 10% Batches Cummulative beta FALT', 'Last 10% Batches Cummulative beta EALT',
                      'Last 10% Batches Cummulative beta EFALT',
                      'Last 10% Batches Cummulative beta EEALT', 'Last 10% Batches Cummulative beta CALT',
                      'Last 10% Batches Cummulative beta AALT'))
    plt.xlabel('Last 10% Batches')
    plt.ylim(top=1)
    plt.title('Number of agents: %i' % numberofagents)
    plt.show()


def rotation(numberofepisodes, numberofagents, waitingPeriodsPerAgent, last10percentwaitingPeriodsPerAgent):
    ####### ROTATION !!!!!!!!!!!!!!
	####### In brief, this function gets the list of Top Agents FOR ALL THE EPISODES, and plots the ROTATION metrics both for the last 10% of episodes and all episodes. 
	####### However last 10% of episaodes was disregarded for the thesis, as it does not give indicative results, by default.
	
    avgWaitingEpisodesPerAgent = numberofagents * [0]
    rotationRatePerAgent = numberofagents * [0]
    finalRotationPerAgent = numberofagents * [0]

    last10percentavgWaitingEpisodesPerAgent = numberofagents * [0]
    last10percentrotationRatePerAgent = numberofagents * [0]
    last10percentfinalRotationPerAgent = numberofagents * [0]

    for i in range(numberofagents):
        if len(waitingPeriodsPerAgent[i]) is 0:
            avgR = 0
        else:
            avgR = sum(waitingPeriodsPerAgent[i]) / len(waitingPeriodsPerAgent[i])

        if avgR > 2 * (numberofagents - 1):
            avgWaitingEpisodesPerAgent[i] = 0
        else:
            avgWaitingEpisodesPerAgent[i] = avgR / (avgR + numberofagents * abs(avgR - (numberofagents - 1)))
        # # print("avgWaitingEpisodesPerAgent[", i, "]=", avgWaitingEpisodesPerAgent[i])

        t = len(waitingPeriodsPerAgent[i])

        if t > 2 * int(numberofepisodes / numberofagents):
            rotationRatePerAgent[i] = 0
        else:
            rotationRatePerAgent[i] = t / (t + numberofagents * abs(t - numberofepisodes / numberofagents))

        # # print("rotationRatePerAgent[", i, "]=", rotationRatePerAgent[i])
        #

        finalRotationPerAgent[i] = (1 * avgWaitingEpisodesPerAgent[i] + 1 * rotationRatePerAgent[i]) / 2
        # # print("finalRotationPerAgent[", i, "]=", finalRotationPerAgent[i])

        for i in range(numberofagents):
            if len(last10percentwaitingPeriodsPerAgent[i]) is 0:
                last10percentavgR = 0
            else:
                last10percentavgR = sum(last10percentwaitingPeriodsPerAgent[i]) / len(
                    last10percentwaitingPeriodsPerAgent[i])

            if last10percentavgR > 2 * (numberofagents - 1):
                last10percentavgWaitingEpisodesPerAgent[i] = 0
            else:
                last10percentavgWaitingEpisodesPerAgent[i] = last10percentavgR / (
                        last10percentavgR + numberofagents * abs(last10percentavgR - (numberofagents - 1)))

            last10percentt = len(last10percentwaitingPeriodsPerAgent[i])

            if last10percentt > 2 * int(numberofepisodes / numberofagents):
                last10percentrotationRatePerAgent[i] = 0
            else:
                last10percentrotationRatePerAgent[i] = last10percentt / (last10percentt + numberofagents * abs(
                    last10percentt - 0.1 * numberofepisodes / numberofagents))

            last10percentfinalRotationPerAgent[i] = (1 * last10percentavgWaitingEpisodesPerAgent[i] + 1 *
                                                     last10percentrotationRatePerAgent[i]) / 2

    print("Final Total Rotation for ", numberofagents, " agents : ",
          sum(finalRotationPerAgent) / len(finalRotationPerAgent))
    labels = []
    for i in range(numberofagents):
        labels.append("Agent " + str(i))

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, avgWaitingEpisodesPerAgent, width, label='avgWaitingEpisodesPerAgent')
    rects2 = ax.bar(x, rotationRatePerAgent, width, label='rotationRatePerAgent')
    rects3 = ax.bar(x + width / 2, finalRotationPerAgent, width, label='finalRotationPerAgent')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylim(0, 1)
    ax.set_ylabel('Rotation')
    ax.set_title('Rotation metric by agent')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = round(rect.get_height(), 3)
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    fig.tight_layout()

    plt.show()

    print("Last 10% episodes' Final Total Rotation for ", numberofagents, " agents : ",
          sum(last10percentfinalRotationPerAgent) / len(last10percentfinalRotationPerAgent))
    last10percentlabels = []
    for i in range(numberofagents):
        last10percentlabels.append("Agent " + str(i))

    x = np.arange(len(last10percentlabels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects4 = ax.bar(x - width / 2, last10percentavgWaitingEpisodesPerAgent, width,
                    label='last10percentavgWaitingEpisodesPerAgent')
    rects5 = ax.bar(x, last10percentrotationRatePerAgent, width, label='last10percentrotationRatePerAgent')
    rects6 = ax.bar(x + width / 2, last10percentfinalRotationPerAgent, width,
                    label='last10percentfinalRotationPerAgent')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylim(0, 1)
    ax.set_ylabel('Last 10% episodes Rotation')
    ax.set_title('Last 10% episodes Rotation metric by agent')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    autolabel(rects4)
    autolabel(rects5)
    autolabel(rects6)

    fig.tight_layout()

    plt.show()


# Create a Class of Agents
class Agent:
    def __init__(self, identifier, numberofagents):
        self.identifier = identifier  # each agent has an index number for identifier, which is a number in this interval: [0,numberofagents)
        self.actionSpace = []  # action 0 for "stay", action 1 for "move forward"
        for i in range(number_of_possible_actions_per_agent):
            self.actionSpace.append(i)
        self.totalReward = 0
        self.number_of_possible_positions_per_agent = number_of_possible_positions_per_agent  # e.g. if there are 3 positions per agent, each agent can either be in 0 (starting block), in 1 (middle block) or in 2 (final block)
        self.action_size = len(self.actionSpace)

        maxpositionsvalues = numberofagents * [number_of_possible_positions_per_agent - 1]
        maxTopPositionAgentsValues = numberofagents * [self.action_size - 1]

        self.state_size = encodeState(maxpositionsvalues, maxTopPositionAgentsValues, numberofagents) + 1

        # self.V={}
        self.Q = np.zeros((self.state_size, self.action_size)) # initialization of thw whole (huge) Q table with zeros

    # Create the Environment!


class Environment:
    def __init__(self, numberofagents):

        self.numberofagents = numberofagents
        self.fullReward = 100  # reward is given in its full only if the winner is only one, otherwise only part of the full reward
        # is equally shared to all the winners. if everyone finishes, they get 0 or penalized .

        # Hyperparameters
        self.alpha = 0.3
        self.gamma = 0.999
        self.epsilon = 0.9
        self.decay = 0.00004

        self.positions = self.numberofagents * [
            0]  # put the agents at the starting point. one cell for each agent (from 0 to numberofpositions-1)
        self.nextpositions = self.numberofagents * [
            0]  # initialize next positions with 0 (they also take values from 0 to numberofpositions-1)
        self.actions = self.numberofagents * [0]  # current actions for each agent
        self.currentTopAgents = self.numberofagents * [0]  # noone is the current winner yet
        self.lastTopAgents = self.numberofagents * [0]  # one cell for each agent (from 0 to 1)
        self.currentrewards = self.numberofagents * [0]  # each cell corresponds to each agent
        self.terminalPositionsPerAgent = self.numberofagents * [
            0]  # this will be needed for fairness metric function below

        self.agents = self.numberofagents * [0]
        for i in range(self.numberofagents):  # create an object list of all agents with i as their identifier
            self.agents[i] = Agent(i, self.numberofagents)

    def Efficiency(self, episodesUntilNow, currentnumberofagents):
        totalRewardOfAllAgents = 0
        for i in range(currentnumberofagents):
            totalRewardOfAllAgents += self.agents[i].totalReward

        return totalRewardOfAllAgents / float(episodesUntilNow * self.fullReward)

    # def Fairness(self):
    #     # print ("Terminal Positions  per agent:", self.terminalPositionsPerAgent)
    #     return min(self.terminalPositionsPerAgent) / float(max(self.terminalPositionsPerAgent))
	
	##################  Simple and Reward fairness were finally estimated within the loop in the "main" function: runExperiment()

    def MultiAgentFairness(self):
        # print ("Terminal Positions  per agent:", self.terminalPositionsPerAgent)
        return min(self.terminalPositionsPerAgent) / float(max(self.terminalPositionsPerAgent))

    # def RewardFairness(self):
    #     # print ("Terminal Positions  per agent:", self.terminalPositionsPerAgent)
    #     return min(self.terminalPositionsPerAgent) / float(max(self.terminalPositionsPerAgent))


## Run the Experiment!


#################### START THE EPISODES ############################    
# print(
#     "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
# print(    "Number of episodes:", numberofepisodes)


def runExperiment(numberofagents):

    env = Environment(numberofagents)
    efficiencyPerEpisode = []
    fairnessPerEpisode = []
    multiFairnessPerEpisode = []
    rewardFairnessPerEpisode = []
    epsilonPerEpisode = []
    countOfsimpleFairnessEpisodesMeasurement = 0
    terminalOccurencesPerEpisode = []
    TopAgentsPerEpisode = []
    UniqueWinnersPerEpisode = []
    waitingEpisodesPerAgent = numberofagents * [0]
    waitingPeriodsPerAgent = []
    last10percentwaitingEpisodesPerAgent = numberofagents * [0]
    last10percentwaitingPeriodsPerAgent = []
	
    for i in range(numberofagents):
        waitingPeriodsPerAgent.append([])
        last10percentwaitingPeriodsPerAgent.append([])

    print(
        "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("Number of Agents:", numberofagents)

    for episode in range(numberofepisodes):  # each episode
        ##print(    "Episode", episode)
        if env.epsilon > 0.004:
            env.epsilon = env.epsilon - env.decay
        else:
            env.epsilon=0

        # print(    "epsilon:", env.epsilon)

        positions = numberofagents * [0]
        nextpositions = numberofagents * [0]
        state = encodeState(numberofagents * [0], numberofagents * [0], numberofagents)
        # nextstate = encodeState(numberofagents * [0], numberofagents * [0], numberofagents)
        env.lastTopAgents = env.currentTopAgents[:]
        env.currentTopAgents = numberofagents * [0]
        env.currentrewards = numberofagents * [0]
        someoneReachedTerminalInThisEpisode = False

        countOfTopAgentsInThisRound = 0
        while not someoneReachedTerminalInThisEpisode:  # it makes new turns/rounds for ever until >=1 agents win(s)

            #################### ALL AGENTS MOVE ##################
            # print(    "Round")

            for i in range(numberofagents):  #### what eachone is going to do in each turns?
                ####  (all agents are considered to move simulteanously)
                ########### EPSILON GREEDY ##############
                if random.uniform(0, 1) < env.epsilon:
                    env.actions[i] = random.choice(env.agents[i].actionSpace)  # Explore action space
                else:
                    env.actions[i] = np.argmax(env.agents[i].Q[state])  # Exploit learned values
                    # print ("not random action:",env.actions[i] )
                # print("actions:", env.actions)

                ############## MOVE AGENT ####################
                if env.actions[i] is 1:

                    nextpositions[i] = nextpositions[
                                           i] + 1  # increment the position of the specific agent, if it decides to move, to update the state

                    if nextpositions[i] is number_of_possible_positions_per_agent - 1:
                        someoneReachedTerminalInThisEpisode = True
                        countOfTopAgentsInThisRound += 1
                        env.terminalPositionsPerAgent[i] += 1
                        env.currentTopAgents[i] = 1  # to update the lastTopAgents later, to prepare the next state
                # print(    "nextpositions:", nextpositions)
            nextstate = encodeState(nextpositions, env.lastTopAgents, numberofagents)
            #### end of turn, every agent made one move! Let them get their Reward now:

            ################### REWARD:

            for i in range(numberofagents):  # let's see who won a reward in this round
                if env.currentTopAgents[i] is 1:  # if this agent reached the top
                    if countOfTopAgentsInThisRound is env.numberofagents:  # if everyone selfishly reached the top
                        env.currentrewards[
                            i] = 0  # they are ALL getting PUNISHED with -1 * fullReward !
                    elif countOfTopAgentsInThisRound > 1:  # if there are more than one winners:
                        env.currentrewards[i] = env.fullReward / pow(numberofagents,
                                                                     2)  # the reward is split accordingly
                    elif countOfTopAgentsInThisRound is 1:  # if there is only one winner:
                        env.currentrewards[i] = env.fullReward  # agent takes the full reward!!!
                else:
                    env.currentrewards[i] = 0  # if this agent did not win in this round, gets 0 reward..

                env.agents[i].totalReward += env.currentrewards[
                    i]  # add the current reward to each agent's sum of rewards variable

            state = encodeState(positions, env.lastTopAgents, numberofagents)

            ############### LEARNING:
            for i in range(numberofagents):
                old_value = env.agents[i].Q[state, env.actions[i]]
                # print(    env.agents[i].Q[nextstate])
                next_max = np.max(env.agents[i].Q[nextstate])

                new_value = (1 - env.alpha) * old_value + env.alpha * (env.currentrewards[i] + env.gamma * next_max)
                env.agents[i].Q[state, env.actions[i]] = new_value

            # RE-INITIALIZE THE GUYS FOR THE NEXT ROUND
            # print(    "rewards:",env.currentrewards,"lastTopAgents",env.lastTopAgents,"prefix:",pow(10,1)*int(''.join(str(y) for y in env.lastTopAgents),2) , "nextpositions", nextpositions, "state", state, "nextstate",nextstate, "done",someoneReachedTerminalInThisEpisode  )
            positions = nextpositions[:] 
            state = nextstate # its just an integer

        ############ END OF ROUNDS

        efficiencyPerEpisode.append(env.Efficiency(episode + 1, numberofagents))
        multiFairnessPerEpisode.append(env.MultiAgentFairness())
        epsilonPerEpisode.append(env.epsilon)

        ######### REWARD FAIRNESS Measurement ##########
        maxRF = env.agents[0].totalReward
        minRF = maxRF
        for bp in range(1, numberofagents):
            if env.agents[bp].totalReward > maxRF:
                maxRF = env.agents[bp].totalReward
            if env.agents[bp].totalReward < minRF:
                minRF = env.agents[bp].totalReward
        if maxRF > 0:
            rewardFairnessPerEpisode.append(minRF / maxRF)
        else:
            rewardFairnessPerEpisode.append(0)
        ####### Simple Fairness Measurement  !!!!!!!!!!!!!!
        if sum(env.currentTopAgents) is 1:
            for i in range(numberofagents):
                if env.currentTopAgents[i] is 1:
                    UniqueWinnersPerEpisode.append(i)
                    break

        if UniqueWinnersPerEpisode:
            maxWinAgent, maxWinningsFreq = Counter(UniqueWinnersPerEpisode).most_common(1)[0]
            minWinAgent = Counter(UniqueWinnersPerEpisode).most_common()[:-2:-1]
            minWinningsFreq = minWinAgent[0][1]
            minWinAgent = minWinAgent[0][0]
            countOfsimpleFairnessEpisodesMeasurement += 1

            if (maxWinAgent is not minWinAgent) and (maxWinningsFreq is not 0):
                fairnessPerEpisode.append(minWinningsFreq / float(maxWinningsFreq))
            else:
                fairnessPerEpisode.append(0)
        ####### ROTATION METRIC #####
        # print("\n env.currentTopAgents", env.currentTopAgents)
        for i in range(numberofagents):
            #             print(    "agent", i)
            if env.currentTopAgents[i] is 0:
                waitingEpisodesPerAgent[i] += 1

                if episode >= numberofepisodes - int(0.1 * numberofepisodes):
                    last10percentwaitingEpisodesPerAgent[i] += 1
            else:
                if waitingEpisodesPerAgent[i] is not 0:
                    waitingPeriodsPerAgent[i].append(waitingEpisodesPerAgent[i])
                    waitingEpisodesPerAgent[i] = 0

                if episode >= numberofepisodes - int(0.1 * numberofepisodes):
                    if last10percentwaitingEpisodesPerAgent[i] is not 0:
                        last10percentwaitingPeriodsPerAgent[i].append(last10percentwaitingEpisodesPerAgent[i])
                        last10percentwaitingEpisodesPerAgent[i] = 0

        ####### LISTS FOR ALTERNATION METRICS !!!!!!!!!!!!!!

        terminalOccurencesPerEpisode.append(sum(env.currentTopAgents))
        TopAgentsPerEpisode.append(env.currentTopAgents)

    ######### END OF EPISODES

    plt.plot(efficiencyPerEpisode, 'r--', multiFairnessPerEpisode, 'b-', epsilonPerEpisode, 'g--', fairnessPerEpisode,
             'b--', rewardFairnessPerEpisode, 'y-')
    plt.gca().legend(('Efficiency', 'Multi-Agent Fairness', 'Epsilon', 'Fairness', 'Reward Fairness'))
    plt.xlabel('Episodes')
    plt.ylim(top=1)
    plt.title('Number of agents: %i' % numberofagents)
    plt.show()

    # PRINT FAIRNESS AND EFFICIENCY
    print("Final Efficiency for ", numberofagents, " agents : ", env.Efficiency(episode + 1, numberofagents))
    if fairnessPerEpisode is not None:
        print("Final (simple) Fairness for ", numberofagents, " agents :", fairnessPerEpisode[-1],
              " Count of Measured Episodes:",
              countOfsimpleFairnessEpisodesMeasurement)
    else:
        print("Final (simple) Fairness for ", numberofagents,
              " agents : Couldn't be calculated. Count of Measured Episodes:",
              countOfsimpleFairnessEpisodesMeasurement)

    print("Final Multi-Agent Fairness for ", numberofagents, " agents :", env.MultiAgentFairness())

    # Histograms of Total Wins, Reaches to the top and Rewards
    sumtotalreward = 0
    for i in range(numberofagents):
        sumtotalreward += env.agents[i].totalReward

    finaltotalrewardsrate = []
    finalwinningsperagent = []
    finalreachestothetopperagent = []
    for i in range(numberofagents):
        finaltotalrewardsrate.append(env.agents[i].totalReward / sumtotalreward)
        finalwinningsperagent.append(UniqueWinnersPerEpisode.count(i) / len(UniqueWinnersPerEpisode))
        finalreachestothetopperagent.append(env.terminalPositionsPerAgent[i] / sum(env.terminalPositionsPerAgent))

    totalRewardFairness = min(finaltotalrewardsrate) / max(finaltotalrewardsrate)
    print("Final Reward Fairness for ", numberofagents, " agents :", totalRewardFairness)

    labels = []
    for i in range(numberofagents):
        labels.append("Agent " + str(i))

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects4 = ax.bar(x - width / 2, finalreachestothetopperagent, width, label='Final Reaches-to-the-Top Rate per agent',
                    color='Blue')
    rects5 = ax.bar(x, finalwinningsperagent, width, label='Final Winnings Rate per agent', color='Green')
    rects6 = ax.bar(x + width / 2, finaltotalrewardsrate, width, label='Total Rewards Rate per agent', color='Yellow')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylim(0, 1)
    ax.set_ylabel('Percentage 0-1')
    ax.set_title('Agents\' Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = round(rect.get_height(), 3)
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects4)
    autolabel(rects5)
    autolabel(rects6)

    fig.tight_layout()

    plt.show()

    alternation(numberofepisodes, numberofagents, terminalOccurencesPerEpisode, TopAgentsPerEpisode)

    rotation(numberofepisodes, numberofagents, waitingPeriodsPerAgent, last10percentwaitingPeriodsPerAgent)


if __name__ == '__main__':
    
	#############Run it in multiprocessing for several max-numbers of agents
	# runExperiment(2)

    import multiprocessing
    from multiprocessing import Process
    from multiprocessing import Pool

    num_processors = number_of_max_agents
    p = Pool(processes=num_processors)
    p.map(runExperiment, [numberofagents for numberofagents in range(2, number_of_max_agents + 1)])
    # for numberofagents in range(2, number_of_max_agents + 1):
    #     p = multiprocessing.Process(target=runExperiment, args=(numberofagents,))
    #     p.start()
    #     p.join()
