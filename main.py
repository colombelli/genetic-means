from genetic_means import GeneticMeans
import pandas as pd
import numpy as np
import random
import multiprocessing as mp

df = pd.read_csv("leukemia_big.csv")
x = df.T.iloc[0:, 1:]

gm = GeneticMeans(df)
gm.evolve()
