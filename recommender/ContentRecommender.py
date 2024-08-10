import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from time import sleep as s
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

global_path = 'recommender/'

class ContentBaseRecommender:

    def __init__(self, data_file= global_path+'world-countries.csv', wait_time=0.1):
        self.data = pd.read_csv(data_file)
        self.wait = wait_time

        self.data = self.process_data(self.data)
        #print(self.data)

        #print('performing vectorization...')
        #s(self.wait)
        
        self.tf_idf = TfidfVectorizer(stop_words='english')
        self.vec = CountVectorizer(stop_words='english')
        
        self.tf_idf_matrix = self.tf_idf.fit_transform(self.data['keywords'])
        self.vec_matrix = self.vec.fit_transform(self.data['keywords'])

        #print('calculating similarity...')
        #s(self.wait)
 
        self.cosine_sim = cosine_similarity(self.tf_idf_matrix, self.tf_idf_matrix)
        self.sim = cosine_similarity(self.vec_matrix, self.vec_matrix)


    def process_data(self, data):
        
        #print('processing data...')
        #s(self.wait)

        def update_keywords(row):
            keywords = str(row['keywords'])
            climate = str(row['climate'])
            if climate == 'mix':
                climate = 'cold hot'
            return keywords + ' ' + climate

        data['keywords'] = data.apply(update_keywords, axis=1)
        data.drop('climate', axis=1, inplace=True)
        data = data.drop_duplicates(subset='Country')
        data['keywords'] = data['keywords'].str.replace(r'\s+', ' ')

        #print('data processed')
        #s(self.wait)
        return data


    def get_TF_IDF_recomendation(self, keywords_matrix, budget, num_of_rec=5):

        #idx = self.data[self.data['Country'].str.lower() == country.lower()].index[0]
        self.cosine_sim = cosine_similarity(keywords_matrix, self.tf_idf_matrix)
        sim_scores = list(enumerate(self.cosine_sim[0]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        #print(sim_scores)
        country_indices = [i[0] for i in sim_scores]

        reced = 0
        recommendation = pd.DataFrame(columns=['ID', 'Country', 'Cost Per Day', 'Similarity'])
        
        for index, score in zip(country_indices, sim_scores):
            #if self.data['Country'].iloc[index].lower() != country.lower():# and self.data['avg cost per day'].iloc[index] <= budget+5:
            recommendation = recommendation._append({'ID': self.data['ID'].iloc[index],
                                                  'Country': self.data['Country'].iloc[index], 
                                                  'Cost Per Day': self.data['avg cost per day'].iloc[index], 
                                                  'Similarity': score[1]}, ignore_index=True)
            reced += 1
        
            if reced == num_of_rec:
                break
        
        return recommendation


    def get_CountVectorizer_recomendation(self, keywords, budget, num_of_rec=5):
        
        #idx = self.data[self.data['Country'].str.lower() == country.lower()].index[0]
        self.sim = cosine_similarity(keywords, self.vec_matrix)

        sim_scores = list(enumerate(self.sim[0]))

        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        #print('score: ',sim_scores)

        country_indices = [i[0] for i in sim_scores]

        reced = 0
        recommendation = pd.DataFrame(columns=['ID', 'Country', 'Cost Per Day', 'Similarity'])
        
        for index, score in zip(country_indices, sim_scores):
            #if self.data['Country'].iloc[index].lower() != country.lower():# and self.data['avg cost per day'].iloc[index] <= budget+5:
            recommendation = recommendation._append({'ID': self.data['ID'].iloc[index],
                                                  'Country': self.data['Country'].iloc[index], 
                                                  'Cost Per Day': self.data['avg cost per day'].iloc[index], 
                                                  'Similarity': score[1]}, ignore_index=True)
            reced += 1
        
            if reced == num_of_rec:
                break
        
        return recommendation
    

    def recommend(self, country, budget=200, num_of_rec=222, tf_idf=True, count_vectorizer=False):

        keywords = []
        for count in country:
            like_df = self.data[self.data['Country'].str.lower() == count.lower()]
            if like_df.empty:
                print(f'Country {count} not found')
            else:
                keywords.append(like_df['keywords'].values[0])
        
        keywords = ' '.join(keywords)
        
        if tf_idf and count_vectorizer:
            tf_idf_matrix = self.tf_idf.transform([keywords])
            vec_matrix = self.vec.transform([keywords])
            rec1 = self.get_TF_IDF_recomendation(tf_idf_matrix, budget, num_of_rec)
            rec2 = self.get_CountVectorizer_recomendation(vec_matrix, budget, num_of_rec)
            return rec1, rec2
        
        elif tf_idf:
            tf_idf_matrix = self.tf_idf.transform([keywords])
            return self.get_TF_IDF_recomendation(tf_idf_matrix, budget, num_of_rec)

        elif count_vectorizer:
            vec_matrix = self.vec.transform([keywords])
            return self.get_CountVectorizer_recomendation(vec_matrix, budget, num_of_rec)


if __name__ == '__main__':
    recommender = ContentBaseRecommender(global_path+'world-countries.csv', .5)

    print(recommender.recommend(['Pakistan', 'India', 'Japan'], count_vectorizer=False))
