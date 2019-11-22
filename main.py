from genetic_means import GeneticMeans
import pandas as pd
import numpy as np
import random
import multiprocessing as mp
from sklearn.cluster import KMeans


if __name__ == '__main__':  
    df = pd.read_csv("leukemia_big.csv", header=None)
    x = df.T.iloc[0:, 1:]
    y = df.iloc[0:1].values[0]

    gm = GeneticMeans(x, y)
    gm.evolve()
