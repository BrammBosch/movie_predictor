import tensorflow as tf
import json
import ast
import numpy as np

from generate_xy import title_to_id, query_imdb, make_input_features

def load_model():
    features = ['genres', 'rating', 'year', 'director']

    new_model = tf.keras.models.load_model('saved_model/genre_cast500_rating_year_director')

    with open('data/genre_cast500_rating_year_director/features.txt') as f:
        feature_str = f.readlines()

    with open('data/genre_cast500_rating_year_director/cast.txt') as f:
        cast_str = f.readlines()


    feature_dict = ast.literal_eval(feature_str[0])
    cast_list = ast.literal_eval(cast_str[0])


    movies = ['Fargo','Interstellar']

    ids = []
    for movie in movies:

        ids.append(title_to_id('Fargo'))

    i=0
    for id in ids:
        print(movies[i])
        calc_result(id,features,cast_list,feature_dict,new_model)
        print("-"*15)
        i+= 1

def calc_result(id,features,cast_list,feature_dict,new_model):
    data = {}
    for feature in features:
        data[feature] = []
    data = query_imdb(id, features, data)
    x = make_input_features(data,cast_list,feature_dict,0)
    #print(x)
    x = np.reshape(x, (1,len(x)))

    prediction = list(new_model.predict(x)[0])
    print(prediction)

    print(prediction.index(max(prediction))+1)

if __name__ == '__main__':
    load_model()