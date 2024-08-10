import pandas as pd
import numpy as np

global_path = 'recommender/'

class PopularityRecommender():
    def __init__(self, dataset=global_path+'world-popularity.csv', alpha=0.5, beta=0.5):
        self.dataset = pd.read_csv(dataset)
        self.alpha = alpha
        self.beta = beta


    def UpdateWeights(self, a, b):
        self.alpha = a
        self.beta = b
    

    def CalculatePopularity(self):
        """
            Popularity Formula:
            Popularity = (((Popularity Index * Avg Visitors) + (Popularity Index * alpha) + (Avg Visitors * beta)) / mean(Popularity Index))/ Max Popularity
        """

        def CalculatePopularityScore(row):

            a = (row['Popularity Index'])*(row['Avg Visitors'])
            a += row['Popularity Index']*self.alpha
            a += row['Avg Visitors']*self.beta 
            a /= np.mean(self.dataset['Popularity Index'])

            return a
        
        self.dataset['Popularity'] = self.dataset.apply(CalculatePopularityScore, axis=1)

        def NormalizePopularity(row):
            return row['Popularity'] / np.max(self.dataset['Popularity'])
        
        self.dataset['Popularity'] = self.dataset.apply(NormalizePopularity, axis=1)


    def recommend(self):
        self.CalculatePopularity()

        return self.dataset[['ID', 'Country', 'Popularity']]


if __name__ == '__main__':
    PR = PopularityRecommender()
    print(PR.Recommend())
