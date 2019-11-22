from genetic_means import GeneticMeans
import pandas as pd


if __name__ == '__main__':  
    df = pd.read_csv("leukemia_big.csv", header=None)
    x = df.T.iloc[0:, 1:]
    y = df.iloc[0:1].values[0]

    gm = GeneticMeans(x, y)
    gm.evolve()
