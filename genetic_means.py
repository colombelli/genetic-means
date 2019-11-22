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
        accuracy = accuracy * -log2(accuracy)
    
    Else:
        accuracy = accuracy * log2(accuracy)

    
    The final fitness function then goes as:
    Fitness = Modified accuracy - Normalized number of genes

"""

import pandas as pd
import numpy as np


class GeneticMeans():
    
    def __init__(df, populationSize=50, iterations=100, mutationRate=0.2, 
                    elitism=0.15, roulette=0.1):

        self.df = df
        self.populationSize = populationSize
        self.iterations = iterations
        self.mutationRate = mutationRate
        self.elitism = elitism
        self.roulette = roulette


    def genetic_algorithm(self)

        self.__generatePopulation()
        self.__computeFitness()


        bestIdx = np.argmax(self.fitness)
        bestIndividualPrev = self.population[bestIdx]
        greaterScoreFound = np.amax(self.fitness)


        generation = 1
        while generation <= max_generations:

            if (self.fitness[bestIdx] > greaterScoreFound):
                greaterScoreFound = np.amax(self.fitness)
                np.savetxt("best_genetic.txt", bestIndividualPrev)

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
        print("\n\nGeneration:", generation)
        print("Best population score:", self.fitness[bestIdx])
        print("Greater score found among generations:", greaterScoreFound)
        print("Accuracy of the best individual:")
        print("Number of genes of the best individual:")
        print("\n\n")


