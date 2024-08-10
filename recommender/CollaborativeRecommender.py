import pandas as pd
import numpy as np
import tez
from sklearn import model_selection, preprocessing
import torch
import torch.nn as nn
import pickle

global_path = 'E:/Projects/Flutter Projects/travel_recommender/lib/python_backend/recommender/'

class RecommenderModel(tez.Model):
    def __init__(self, num_users, num_country, lr=1e-3):
        super().__init__()

        self.learning_rate = lr
        self.user_embed = nn.Embedding(num_users, 64)   
        self.country_embed = nn.Embedding(num_country, 64)
        
        self.hidden1 = nn.Linear(128, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.hidden3 = nn.Linear(128, 128)

        self.out = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.step_scheduler_after = 'epoch'

    def moniter_metrics(self, outputs, targets):
        outputs = outputs.cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()
        return {'rmse': (np.sqrt(((outputs - targets) ** 2).mean()))}

    def forward(self, user, country, rating=None):
        
        user = self.user_embed(user)
        country = self.country_embed(country)
        out = torch.cat([user, country], 1)
        
        out = self.hidden1(out)
        out = self.relu(out)

        out = self.hidden2(out)
        out = self.relu(out)

        out = self.hidden3(out)
        out = self.relu(out)

        out = self.out(out)
        
        if rating is None:
            return out
        
        loss = nn.MSELoss()(out, rating.view(-1, 1))
        cal_metrics = self.moniter_metrics(out, rating.view(-1, 1))

        return out, loss, cal_metrics
    
    def fetch_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def fetch_scheduler(self):
        return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.7)


class Dataset:
    def __init__(self, user, country, rating):
        self.user = user
        self.country = country
        self.rating = rating
    
    def __len__(self):
        return len(self.user)
    
    def __getitem__(self, item):
        return {
            'user': torch.tensor(self.user[item], dtype=torch.long),
            'country': torch.tensor(self.country[item], dtype=torch.long),
            'rating': torch.tensor(self.rating[item], dtype=torch.float)
        }


def train_NN(dataset_name, model_name, epochs=20):
    
    df = pd.read_csv(dataset_name)
    
    lbl_user = preprocessing.LabelEncoder()
    lbl_country = preprocessing.LabelEncoder()

    df.user = lbl_user.fit_transform(df.user.values)
    df.country = lbl_country.fit_transform(df.country.values)

    train_df, test_df = model_selection.train_test_split(df, test_size=0.2, random_state=42, stratify=df.rating.values)

    train_dataset = Dataset(user=train_df.user.values, country=train_df.country.values, rating=train_df.rating.values)

    test_dataset = Dataset(user=test_df.user.values, country=test_df.country.values, rating=test_df.rating.values)

    #es = EarlyStopping(monitor="valid_loss", model_path="model.bin")

    model = RecommenderModel(num_users=len(lbl_user.classes_), num_country=len(lbl_country.classes_), lr=1e-2)
    model.fit(train_dataset, test_dataset, train_bs=1000, valid_bs=1000, fp16=False, epochs=epochs, device='cpu')

    print('model trained')

    input('press any key to save the model')

    model.save(global_path+'model' + model_name)

    with open(global_path+'user_encoder.pkl', 'wb') as f:
        pickle.dump(lbl_user, f)
    with open(global_path+'country_encoder.pkl', 'wb') as f:
        pickle.dump(lbl_country, f)


class CollaborativeRecommender:
    def __init__(self, user: int, model_name):
        self.user = user
        self.model_name = model_name

    def updateModel(self, dataset_name=None, epochs=1):
        df = pd.read_csv(dataset_name)
    
        # loading encoders used during training
        with open(global_path+'user_encoder.pkl', 'rb') as f:
            lbl_user = pickle.load(f)
        with open(global_path+'country_encoder.pkl', 'rb') as f:
            lbl_country = pickle.load(f)

        # loading pre_trained model
        model = RecommenderModel(num_users=len(lbl_user.classes_), num_country=len(lbl_country.classes_))
        model.load(global_path+ 'model/' + self.model_name, device='cpu')

        train_df, test_df = model_selection.train_test_split(df, test_size=0.2, random_state=42, stratify=df.rating.values)

        train_dataset = Dataset(user=train_df.user.values, country=train_df.country.values, rating=train_df.rating.values)

        test_dataset = Dataset(user=test_df.user.values, country=test_df.country.values, rating=test_df.rating.values)

        model.fit(train_dataset, test_dataset, train_bs=1000, valid_bs=1000, fp16=False, epochs=epochs, device='cpu')

        print('model trained')

        input('press any key to save the model')

        model.save(global_path+'model/' + self.model_name)

        with open(global_path+'user_encoder.pkl', 'wb') as f:
            pickle.dump(lbl_user, f)
        with open(global_path+'country_encoder.pkl', 'wb') as f:
            pickle.dump(lbl_country, f)

    def recommend(self, test=False,  top_n: int=222):

        # loading encoders used during training
        with open(global_path+'user_encoder.pkl', 'rb') as f:
            lbl_user = pickle.load(f)
        with open(global_path+'country_encoder.pkl', 'rb') as f:
            lbl_country = pickle.load(f)

        # loading pre_trained model
        pre_trained_model = RecommenderModel(num_users=len(lbl_user.classes_), num_country=len(lbl_country.classes_))
        pre_trained_model.load(global_path+'model/' + self.model_name, device='cpu')

        user_id = lbl_user.transform([self.user])[0]

        user_id = torch.tensor([user_id] * len(lbl_country.classes_)).long()
        country_ids = torch.tensor(range(len(lbl_country.classes_))).long()

        
        with torch.no_grad():
            predictions = pre_trained_model(user_id, country_ids)
        
        predictions = predictions.flatten()

        recommendation = pd.DataFrame({'ID': country_ids.numpy(), 'Rating': predictions.numpy()})

        df = pd.read_csv(global_path+'world-countries.csv')

        recommendation = pd.merge(recommendation, df, how='left', left_on='ID', right_on='ID')
        
        if test:
            recommendation = recommendation.sort_values(by='Rating', ascending=False).head(top_n)


        return recommendation[['ID', 'Country', 'Rating']]

    

def Recommender(user, model_name, top_n=10, train=False, dataset_name=None):
    
    if train:
        if dataset_name is None:
            raise ValueError('Please provide dataset_name for trainig the model')
        else:
            train_NN(dataset_name, model_name)

    # loading encoders used during training
    with open(global_path+'user_encoder.pkl', 'rb') as f:
        lbl_user = pickle.load(f)
    with open(global_path+'country_encoder.pkl', 'rb') as f:
        lbl_country = pickle.load(f)

    # loading pre_trained model
    pre_trained_model = RecommenderModel(num_users=len(lbl_user.classes_), num_country=len(lbl_country.classes_))
    pre_trained_model.load(global_path+'model/' + model_name, device='cpu')

    user_id = lbl_user.transform([user])[0]

    user_id = torch.tensor([user_id] * len(lbl_country.classes_)).long()
    country_ids = torch.tensor(range(len(lbl_country.classes_))).long()

    
    with torch.no_grad():
        predictions = (pre_trained_model(user_id, country_ids))
    

    # Get the top N country IDs
    top_n_country_ids = predictions.flatten().argsort(descending=True)[:top_n]

    return top_n_country_ids


if __name__ == "__main__":

    model = CollaborativeRecommender(user=0, model_name='CF_Neural_Model3.7.bin')
    print(model.recommend(test=True, top_n=16))

    model.updateModel(dataset_name=global_path+'ratings.csv', epochs=1)