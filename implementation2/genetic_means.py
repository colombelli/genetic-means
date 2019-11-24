"""

@Title: Genetic Means
@Author: Felipe Colombelli
@Description: A genetic algorithm using k-means as classification model
              for selecting genes out of a dataset with two types of 
              leukemia: ALL and AML.




* Chromosome encoding:

    An 1D array with zeros and ones representing what gene is being considered.
    e.g.
        Lets consider 5 genes;
        The 2nd and 4th are being considered;
        The corresponding chromosome would be represented by the array:
        [0, 1, 0, 1, 0]




* Fitness function rationale:

    Because our objective is to find the minimum amount of genes that explain our
    data, we will use a fitness function based on the model accuracy AND the number
    of genes.

    We will start our rationale from the following idea:
    Fitness = Accuracy - Number of genes

    Because the number of genes is ranged from 0 to 7128, the number of genes would
    take much more importance, so we normalize it mapping the values to range between
    0 and 100 using the following formula: 

    100 * (number of genes - min) / (max - min), where min = 0, and max = 7128
    100 * (number of genes) / 7128

    Now, Fitness = Accuracy - Normalized number of genes

    We still want our model to prioritize the accuracy. Lets assume, for instance,
    that an accuracy below or equal to 90% is utterly trash. Indeed, this claim is 
    based on the 98.6% accuracy got from the model using all the features.
    To treat those accuracies as bad configurations, we will penalize them shifting
    the numbers to the interval [-100, -10]. 

    If accuracy < 90:
        accuracy = accuracy - 100

    Finally, we will boost up gains in accuracy to tell the algorithm that even if it 
    could reduce features, just do so by means of the Computer Science magic scale: log2.
    In other words, we will tell that gains in accuracy are log2 more valuable.

    If accuracy < 90:
        accuracy = accuracy - 100
        accuracy = accuracy * -log2(-accuracy)
    
    Else:
        accuracy = accuracy * log2(accuracy)

    
    The final fitness function then goes as:
    Fitness = Modified accuracy - Normalized number of genes

"""

import pandas as pd
import numpy as np
import random
import multiprocessing as mp
from sklearn.cluster import KMeans
from math import log2
import pickle
import csv


NUM_OF_GENES = 7128

class GeneticMeans():
    
    def __init__(self, df, dfLabels, populationSize=50, iterations=1000, 
                    mutationRate=0.001, elitism=0.3):

        self.df = df
        self.dfLabels = dfLabels
        self.populationSize = populationSize
        self.iterations = iterations
        self.mutationRate = mutationRate
        self.elitism = elitism

        self.fitness = []
        self.population = []

    def evolve(self):

        self.__generatePopulation()
        self.__computeFitness()

        bestIdx = np.argmax(self.fitness)
        bestIndividualPrev = self.population[bestIdx]
        greaterScoreFound = np.amax(self.fitness)


        generation = 1
        while generation <= self.iterations:

            if (self.fitness[bestIdx] > greaterScoreFound):
                greaterScoreFound = np.amax(self.fitness)
                np.savetxt("best_genetic.txt", bestIndividualPrev)

            self.__printIterationStatus(generation, bestIdx, greaterScoreFound)
            self.__selectPopulation()
            self.__crossPopulation()
            self.__computeFitness()
            generation += 1

            bestIdx = np.argmax(self.fitness)
            bestIndividual = self.population[bestIdx]
            bestIndividualPrev = bestIndividual

        print("Max generations reached. Learning algorithm stopped.")
        return



    def __printIterationStatus(self, generation, bestIdx, greaterScoreFound):
       
        bestIndividual = self.population[bestIdx]
        bestScore = self.fitness[bestIdx]
        bestAccuracy = self.calculateAccuracy(bestIndividual)
        numGenesBestIndividual = np.sum(bestIndividual)

        print("\n\nGeneration:", generation)
        print("Best score among the population:", bestScore)
        print("Greater score found among generations:", greaterScoreFound)
        print("Accuracy of the best individual:", bestAccuracy)
        print("Number of genes of the best individual:", numGenesBestIndividual)
        print("\n\n")

        self.__dumpResults(generation, bestIndividual, bestScore, bestAccuracy, numGenesBestIndividual)
        return

    
    def __dumpResults(self, generation, bestIndividual, bestScore, bestAccuracy, numGenesBestIndividual):
        
        with open('ga_pop.pkl', 'wb') as pop_file:
            pickle.dump(self.population, pop_file)

        with open('ga_best_individual.pkl', 'wb') as best:
            pickle.dump(bestIndividual, best)

        with open('ga_info.csv', "a", newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow([generation, bestScore, bestAccuracy, numGenesBestIndividual])



    def __generatePopulation(self):

        population = np.full((self.populationSize, NUM_OF_GENES), False, dtype=bool)
        turningProcedures = 5
        for turns in range(turningProcedures):
            population = self.__selectInitialGenes(population)

        self.population = population
        return

    def __selectInitialGenes(self, population):
        
        initialSelectedGeneForIndividual = np.random.randint(low=0, high=NUM_OF_GENES, size=self.populationSize)
        for idx, individual in enumerate(population):
            individual[initialSelectedGeneForIndividual[idx]] = True
        return population



    def __computeFitness(self):

        self.fitness = [None] * len(self.population)
        pool = mp.Pool(mp.cpu_count())

        self.fitness = np.array([pool.apply(self.computeIndividualFitness, args=(individual, ))
                   for individual in self.population])
        
        pool.close()
        return


    # The three next methods (and also another ones from this file) 
    # violate the information hiding principles because of pickling 
    # issues bothering the parallel computation
    def computeIndividualFitness(self, individual):
        
        accuracy = self.calculateAccuracy(individual)

        if accuracy < 90:
            accuracy = accuracy - 100
            accuracy = -log2(-accuracy) * accuracy
        else:
            accuracy = accuracy * log2(accuracy)
        
        numOfSelectedGenes = np.sum(np.array(individual))
        normNumOfSelectedGenes = 100 * (numOfSelectedGenes) / 7128

        fitness = accuracy - normNumOfSelectedGenes
        return fitness   

    def calculateAccuracy(self, individual):

        reducedDF = self.df[self.df.columns[individual]]
        if len(reducedDF.columns) == 0:
            return 0
        
        kmeans = KMeans(n_clusters=2, n_init=50)
        kmeans.fit(reducedDF)

        predictedLabels = kmeans.predict(reducedDF)

        # Because there's no way to know which cluster will be assigned to each class
        realLabels_01 = self.convertLabelsTo01(0, 1)
        realLabels_10 = self.convertLabelsTo01(1, 0)

        rigthGuesses01 = (np.array(realLabels_01) == predictedLabels)
        rigthGuesses10 = (np.array(realLabels_10) == predictedLabels)
        rigthGuesses = max(np.sum(rigthGuesses01), np.sum(rigthGuesses10))

        numSamples = len(self.dfLabels) 
        numRigthGuesses = np.sum(rigthGuesses)

        accuracy = numRigthGuesses / numSamples * 100

        return accuracy

    def convertLabelsTo01(self, ALL, AML):
        
        realLabels_01 = []  
        for label in list(self.dfLabels):
            if label == 'ALL':
                realLabels_01.append(ALL)
            elif label == 'AML':
                realLabels_01.append(AML)
        return realLabels_01



    def __selectPopulation(self):

        numElite = round(self.elitism * self.populationSize)
        # Get the index of the N greatest scores:
        eliteIdx = np.argpartition(self.fitness, -numElite)[-numElite:]
        elite = self.population[eliteIdx]

        self.population = elite
        return



    def __crossPopulation(self):

        missingPopulation = []
        numMissingIndividuals = self.populationSize - len(self.population)

        mask = np.random.randint(0, 2, size=self.population.shape[1])
        # mask example for a problem with 5 genes [0,1,1,0,1]
        # meaning that dad0 passes its first gene, da1 its second, and so on...

        for _ in range(numMissingIndividuals):
            dad0Idx = np.random.randint(0, len(self.population))
            dad1Idx = np.random.randint(0, len(self.population))
            dad0 = self.population[dad0Idx]
            dad1 = self.population[dad1Idx]
            son = []

            for i, gene in enumerate(mask):
                if gene == 0:
                    son.append(dad0[i])
                else:
                    son.append(dad1[i])

            son = np.array(son)
            missingPopulation.append(son)


        missingPopulation = np.array(missingPopulation)
        missingPopulation = self.__mutatePopulation(missingPopulation)
        self.population = np.append(self.population, missingPopulation, axis=0)
        return


    def __mutatePopulation(self, missingPopulation):

        mutationPercentage = self.mutationRate*1000000
        
        for individual in missingPopulation:
            for gene in range(len(individual)):
                
                rand = np.random.randint(0, 1000001)
                if self.__mustMutate(rand, mutationPercentage):
                    individual[gene] = ~individual[gene]

        return missingPopulation

    def __mustMutate(self, rand, mutation):
        return rand <= mutation