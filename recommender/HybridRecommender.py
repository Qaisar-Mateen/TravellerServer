from recommender.PopularityRecommender import PopularityRecommender
from recommender.CollaborativeRecommender import CollaborativeRecommender
from recommender.ContentRecommender import ContentBaseRecommender
import pandas as pd

class HybridRecommender:
    def __init__(self, popularity_model: PopularityRecommender=False,
                collaborative_model: CollaborativeRecommender=False,
                content_model: ContentBaseRecommender=False,
                popular_weight: float=0.2, collab_weight: float=0.5, content_weight: float=0.4):
        
        if popularity_model:
            self.popularity_model = PopularityRecommender()

        else:
            popularity_model = None
        
        if collaborative_model[0]:
            try:
                self.collaborative_model = CollaborativeRecommender(user=collaborative_model[1], model_name=collaborative_model[2])
            except:
                raise TypeError('Collaborative Model must be a tuple (preset:bool, user:int, model_name:str)')
        else:
            self.collaborative_model = None

        if content_model:
            self.content_model = ContentBaseRecommender(wait_time=0)
        else:
            self.content_model = None
        
        self.alpha = popular_weight
        self.beta = collab_weight
        self.gamma = content_weight

    def recommend(self, top_n=222):

        if self.popularity_model is not None:
            popularity_recs = self.popularity_model.recommend()

        if self.collaborative_model is not None:
            collaborative_recs = self.collaborative_model.recommend()

        if self.content_model is not None and self.collaborative_model is not None:

            country = collaborative_recs.sort_values('Rating', ascending=False)['Country'].head(5).values
            country = list(map(str, country))
            
            content_recs = self.content_model.recommend(country=country)


        if self.content_model is None:
            recommendations = pd.merge(popularity_recs, collaborative_recs, on=('ID', 'Country'), how='outer')

            recommendations['Score'] = recommendations['Popularity'] * self.alpha + recommendations['Rating'] * self.beta
                
        elif self.collaborative_model is None:
            recommendations = pd.merge(popularity_recs, content_recs, on=('ID', 'Country'), how='outer')

            recommendations['Score'] = recommendations['Popularity'] * self.alpha + recommendations['Similarity'] * self.gamma
                
        else:
            recommendations = pd.merge(popularity_recs, collaborative_recs, on=('ID', 'Country'), how='outer')
            recommendations = pd.merge(recommendations, content_recs, on=('ID', 'Country'), how='outer')

            recommendations['Score'] = recommendations['Popularity']*self.alpha + recommendations['Rating']*self.beta + recommendations['Similarity']*self.gamma


        recommendations = recommendations.sort_values(by='Score', ascending=False)

        return recommendations[['ID', 'Country', 'Score']][:top_n]
    

if __name__ == '__main__':

    hr = HybridRecommender(collaborative_model=(True, 0, 'CF_Neural_Model3.7.bin'),
                        popularity_model=True,
                        popular_weight=0.2, collab_weight=0.8)
    print(hr.recommend(top_n=16))
    print()

    hr2 = HybridRecommender(collaborative_model=(True, 0, 'CF_Neural_Model3.7.bin'),
                popularity_model=True, content_model=True,
                popular_weight=0.15, collab_weight=0.7, content_weight=0.15
            )
    
    print(hr2.recommend(top_n=16))
