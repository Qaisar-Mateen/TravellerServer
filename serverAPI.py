import flask
import pandas as pd
from flask import request
import recommender.HybridRecommender as recommender
from recommender.PopularityRecommender import PopularityRecommender

app = flask.Flask(__name__)

@app.route("/login", methods=["GET"])
def login():
    userId = request.args.get("userId")
    if userId is None:
        return {'valid': False}, 400
    userId = int(userId)
    ids = pd.read_csv('recommender/ratings.csv')
    ids = ids['user'].unique()

    if userId in ids:
        return {"valid": True}
    return {"valid": False}, 400


@app.route("/recommend", methods=["GET"])
def recommend():
    userId = request.args.get("userId")
    
    if "popular" in request.args:
        hr = PopularityRecommender()
        recommendations =  hr.recommend()
        recommendations = recommendations.sort_values(by='Popularity', ascending=False)
        recommendations = recommendations.head(20)
        return recommendations.to_json(orient='records')
    
    elif userId is None:
        return {'valid': False}, 400

    else:
        hr = recommender.HybridRecommender(collaborative_model=(True, userId, 'CF_Neural_Model3.7.bin'),
            popularity_model=True, content_model=True,
            popular_weight=0.15, collab_weight=0.7, content_weight=0.15
        )

        recommendations = hr.recommend()
        return recommendations.to_json(orient='records')
    
@app.route("/recommend/cities", methods=["GET"])
def getCities():
    country = request.args.get("country")
    if country is None:
        return {'valid': False}, 400

    cities = pd.read_csv('recommender/world-cities.csv')
    cities = cities[cities['country'] == country][['name', 'lat', 'lng']]

    if len(cities) > 4:
        top_three = cities.iloc[:3]
            
        remaining = cities.iloc[3:]
        remaining = remaining.sample(n=min(5, len(remaining)))
        cities = pd.concat([top_three, remaining])
        
    return cities.to_json(orient='records') 

if __name__ == "__main__":
    app.run(debug= True, host="0.0.0.0", port=5000)
