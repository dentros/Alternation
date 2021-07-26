import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd

def alternation(numberofepisodes, numberofagents, terminalOccurencesPerEpisode, TopAgentsPerEpisode):
    ####### ALTERNATION METRICS !!!!!!!!!!!!!!
    numberOfBatches = numberofepisodes - (numberofagents - 1)
    # print("Number of batches", numberOfBatches)

    termOccPerBatch = numberOfBatches * [0]
    numOfWinnersPerBatch = numberOfBatches * [0]

    betaFALT = numberOfBatches * [0]
    betaEALT = numberOfBatches * [0]
    betaEFALT = numberOfBatches * [0]
    betaEEALT = numberOfBatches * [0]
    betaCALT = numberOfBatches * [0]
    betaAALT = numberOfBatches * [0]

    for batchId in range(0, numberOfBatches):
        #         print(    "BATCH: ", batchId)

        whoReachedTopInThisBatch = numberofagents * [
            0]  # one element per episode of the batch. It calculates how many agents managed to reach the top at each episode
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

        #      beta values for each batch:

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

        betaAALT[batchId] = countOfExclusiveWinnersInBatch / numOfWinnersPerBatch[batchId]

    return (sum(betaFALT) / numberOfBatches), (sum(betaEALT) / numberOfBatches), (sum(betaEFALT) / numberOfBatches), (
                sum(betaEEALT) / numberOfBatches), (sum(betaCALT) / numberOfBatches), (sum(betaAALT) / numberOfBatches)


def rotation(numberofagents, numberofepisodes, waitingPeriodsPerAgent):
    ####### ROTATION !!!!!!!!!!!!!!
    avgWaitingEpisodesPerAgent = numberofagents * [0]
    rotationRatePerAgent = numberofagents * [0]
    finalRotationPerAgent = numberofagents * [0]

    for i in range(numberofagents):

        if len(waitingPeriodsPerAgent[i]) is 0:
            avgR = 0
        else:
            avgR = sum(waitingPeriodsPerAgent[i]) / len(waitingPeriodsPerAgent[i])

        avgWaitingEpisodesPerAgent[i] = avgR / (
                    avgR + numberofagents * numberofagents * abs(avgR - (numberofagents - 1)))

        t = len(waitingPeriodsPerAgent[i])

        rotationRatePerAgent[i] = t / (t + numberofagents * numberofagents * abs(t - numberofepisodes / numberofagents))

        finalRotationPerAgent[i] = (1 * avgWaitingEpisodesPerAgent[i] + 1 * rotationRatePerAgent[i]) / 2
        # print("finalRotationPerAgent[", i, "]=", finalRotationPerAgent[i])
    return sum(finalRotationPerAgent) / len(finalRotationPerAgent)


def Efficiency(episodes, currentnumberofagents, totalRewardperagent, fullReward):
    totalRewardOfAllAgents = 0
    for i in range(currentnumberofagents):
        totalRewardOfAllAgents += totalRewardperagent[i]

    return totalRewardOfAllAgents / float(episodes * fullReward)


MaxAgentsLimit = 10

numberofepisodes = 1000

resultsALT = []
resultsALT.append([])
resultsALT.append([])

resultsROT = []
resultsROT.append([])
resultsROT.append([])

resultsEFF = []
resultsEFF.append([])
resultsEFF.append([])

resultsMULTIFAIR = []
resultsMULTIFAIR.append([])
resultsMULTIFAIR.append([])

resultsFAIR = []
resultsFAIR.append([])
resultsFAIR.append([])

resultsREWFAIR = []
resultsREWFAIR.append([])
resultsREWFAIR.append([])

ratioVsFALTmetric = []
ratioVsEALTmetric = []
ratioVsEFALTmetric = []
ratioVsEEALTmetric = []
ratioVsCALTmetric = []
ratioVsAALTmetric = []

ratioVsROTmetric = []

ratioVsEFFmetric = []
ratioVsFAIRmetric = []
ratioVsMULTIFAIRmetric = []
ratioVsREWFAIRmetric = []

for numberofagents in range(2, MaxAgentsLimit):
    resultsALT.append((numberofagents + 1) * [(0, 0, 0, 0, 0, 0)])
    resultsROT.append((numberofagents + 1) * [0])
    resultsEFF.append((numberofagents + 1) * [0])
    resultsFAIR.append((numberofagents + 1) * [0])
    resultsMULTIFAIR.append((numberofagents + 1) * [0])
    resultsREWFAIR.append((numberofagents + 1) * [0])

    for selectedNumberOfAgents in range(1, numberofagents + 1):
        # print("£££££ NUMBER OF AGENTS :", numberofagents , " Rotating Agents:", selectedNumberOfAgents, " £££££" )

        TopAgentsPerEpisode = []
        terminalOccurencesPerEpisode = numberofepisodes * [0]

        for i in range(numberofepisodes):
            TopAgentsPerEpisode.append(numberofagents * [0])

        for i in range(numberofepisodes):
            # TopAgentsPerEpisode[i][i % numberofagents] = 1 # Perfect Alternation
            # TopAgentsPerEpisode[i][i % 2] = 1 # Only 2 agents rotate
            TopAgentsPerEpisode[i][i % selectedNumberOfAgents] = 1  # Only n/2 agents rotate

        # print(TopAgentsPerEpisode)

        for i in range(numberofepisodes):
            terminalOccurencesPerEpisode[i] = sum(TopAgentsPerEpisode[i])

        resultsALT[numberofagents][selectedNumberOfAgents] = alternation(numberofepisodes, numberofagents,
                                                                         terminalOccurencesPerEpisode,
                                                                         TopAgentsPerEpisode)

        waitingPeriodsPerAgent = numberofagents * [0]

        for i in range(numberofagents):
            if i < selectedNumberOfAgents:
                waitingPeriodsPerAgent[i] = int(numberofepisodes / numberofagents) * [numberofagents - 1]
            else:
                waitingPeriodsPerAgent[i] = []

        # print("waitingPeriodsPerAgent", waitingPeriodsPerAgent)

        # print("resultsROT:",resultsROT )

        fullreward = 100
        totalRewPerAgent = numberofagents * [0]
        for s in range(selectedNumberOfAgents):
            totalRewPerAgent[s] = fullreward * numberofepisodes / numberofagents

        resultsEFF[numberofagents][selectedNumberOfAgents] = (Efficiency(numberofepisodes, numberofagents,
                                                                         totalRewPerAgent, fullreward) +
                                                              resultsROT[numberofagents][selectedNumberOfAgents]) / 2

        resultsROT[numberofagents][selectedNumberOfAgents] = rotation(numberofagents, numberofepisodes,
                                                                      waitingPeriodsPerAgent) * Efficiency(
            numberofepisodes, numberofagents, totalRewPerAgent, fullreward)

        ratioVsFALTmetric.append(
            [selectedNumberOfAgents / numberofagents, resultsALT[numberofagents][selectedNumberOfAgents][0]])
        ratioVsEALTmetric.append(
            [selectedNumberOfAgents / numberofagents, resultsALT[numberofagents][selectedNumberOfAgents][1]])
        ratioVsEFALTmetric.append(
            [selectedNumberOfAgents / numberofagents, resultsALT[numberofagents][selectedNumberOfAgents][2]])
        ratioVsEEALTmetric.append(
            [selectedNumberOfAgents / numberofagents, resultsALT[numberofagents][selectedNumberOfAgents][3]])
        ratioVsCALTmetric.append(
            [selectedNumberOfAgents / numberofagents, resultsALT[numberofagents][selectedNumberOfAgents][4]])
        ratioVsAALTmetric.append(
            [selectedNumberOfAgents / numberofagents, resultsALT[numberofagents][selectedNumberOfAgents][5]])

        ratioVsROTmetric.append(
            [selectedNumberOfAgents / numberofagents, resultsROT[numberofagents][selectedNumberOfAgents]])

        ratioVsEFFmetric.append(
            [selectedNumberOfAgents / numberofagents, resultsEFF[numberofagents][selectedNumberOfAgents]])

    # print ("ResultsALT:", resultsALT)

ratioVsFALTmetric.append([0, 0])
ratioVsEALTmetric.append([0, 0])
ratioVsEFALTmetric.append([0, 0])
ratioVsEEALTmetric.append([0, 0])
ratioVsCALTmetric.append([0, 0])
ratioVsAALTmetric.append([0, 0])

ratioVsROTmetric.append([0, 0])

ratioVsEFFmetric.append([0, 0])

ratioVsFALTmetric.sort()
ratioVsEALTmetric.sort()
ratioVsEFALTmetric.sort()
ratioVsEEALTmetric.sort()
ratioVsCALTmetric.sort()
ratioVsAALTmetric.sort()

ratioVsROTmetric.sort()

ratioVsEFFmetric.sort()

# print("ratioVsFALTmetric: ", ratioVsFALTmetric)
# print("ratioVsEALTmetric: ", ratioVsEALTmetric)
# print("ratioVsEFALTmetric: ", ratioVsEFALTmetric)
# print("ratioVsEEALTmetric: ", ratioVsEEALTmetric)
# print("ratioVsCALTmetric: ", ratioVsCALTmetric)
# print("ratioVsAALTmetric: ", ratioVsAALTmetric)




#


ratioVsFALTmetric = pd.DataFrame(ratioVsFALTmetric, columns=["Ratio", "FALT"
                                                             ])
ratioVsEALTmetric = pd.DataFrame(ratioVsEALTmetric, columns=["Ratio", "EALT"
                                                             ])
ratioVsEFALTmetric = pd.DataFrame(ratioVsEFALTmetric, columns=["Ratio", "EFALT"
                                                               ])
ratioVsEEALTmetric = pd.DataFrame(ratioVsEEALTmetric, columns=["Ratio", "EEALT"
                                                               ])
ratioVsCALTmetric = pd.DataFrame(ratioVsCALTmetric, columns=["Ratio", "CALT"
                                                             ])
ratioVsAALTmetric = pd.DataFrame(ratioVsAALTmetric, columns=["Ratio", "AALT"
                                                             ])

ratioVsROTmetric = pd.DataFrame(ratioVsROTmetric, columns=["Ratio", "Rotation*Efficiency"
                                                           ])

ratioVsEFFmetric = pd.DataFrame(ratioVsEFFmetric, columns=["Ratio", "(Efficiency+Rot)/2"  ])


modelF = LinearRegression()


xf = ratioVsFALTmetric.iloc[:, 0].tolist()
xf = np.array(xf).reshape((-1, 1))
yf = ratioVsFALTmetric.iloc[:, 1].tolist()
yf = np.array(yf)
modelF.fit(xf, yf)
print('FALT intercept b0:', modelF.intercept_)
print('FALT slope b1:', modelF.coef_)

modelE = LinearRegression()
xe = ratioVsEALTmetric.iloc[:, 0].tolist()
xe = np.array(xe).reshape((-1, 1))
ye = ratioVsEALTmetric.iloc[:, 1].tolist()
ye = np.array(ye)

modelE.fit(xe, ye)
print('EALT intercept b0:', modelE.intercept_)
print('EALT slope b1:', modelE.coef_)

modelR = LinearRegression()
xr = ratioVsROTmetric.iloc[:, 0].tolist()
xr = np.array(xr).reshape((-1, 1))
yr = ratioVsROTmetric.iloc[:, 1].tolist()
yr = np.array(yr)

modelR.fit(xr, yr)
print('ROT intercept b0:', modelR.intercept_)
print('ROT slope b1:', modelR.coef_)



xEF = np.array(ratioVsEFALTmetric.iloc[:, 0].tolist(), dtype=float)
yEF = np.array(ratioVsEFALTmetric.iloc[:, 1].tolist(), dtype=float)

curve_fit = np.polyfit(yEF, xEF, 1)
print("EFALT e^b0  + e^(b1*x). Coefficients b0, b1:", curve_fit)
plt.plot(xEF, yEF, "o")

EFregression = np.exp(curve_fit[0]) * np.exp(curve_fit[1]*xEF)
plt.plot(xEF,EFregression)






# print(ratioVsCALTmetric)
ratioVsFALTmetric.plot(x="Ratio", y="FALT", kind="scatter", figsize=(10, 5), grid=True)
ratioVsEALTmetric.plot(x="Ratio", y="EALT", kind="scatter", figsize=(10, 5), grid=True)
ratioVsEFALTmetric.plot(x="Ratio", y="EFALT", kind="scatter", figsize=(10, 5), grid=True)
ratioVsEEALTmetric.plot(x="Ratio", y="EEALT", kind="scatter", figsize=(10, 5), grid=True)
ratioVsCALTmetric.plot(x="Ratio", y="CALT", kind="scatter", figsize=(10, 5), grid=True)
ratioVsAALTmetric.plot(x="Ratio", y="AALT", kind="scatter", figsize=(10, 5), grid=True)

ratioVsROTmetric.plot(x="Ratio", y="Rotation*Efficiency", kind="scatter", figsize=(10, 5), grid=True)

ratioVsEFFmetric.plot(x="Ratio", y="(Efficiency+Rot)/2", kind="scatter", figsize=(10, 5), grid=True)
plt.show()
