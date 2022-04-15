from itertools import islice

from imdb import IMDb
import requests
import csv
from collections import OrderedDict

def generate_xy_main(features, n_actors, data_file):

    list_id = read_excel(data_file)

    data = {}

    for feature in features:
        data[feature] = []

    for i, id in enumerate(list_id):
        print(i)
        data = query_imdb(id[0], features,data)

    cast_list, feature_dict = vector_features(data, features, n_actors)

    textfile_features = open("data/genre_cast500_rating_year_director/features.txt", "w")
    textfile_cast = open("data/genre_cast500_rating_year_director/cast.txt", "w")
    textfile_features.write(str(feature_dict))
    textfile_cast.write(str(cast_list))

    X = []
    Y = []

    for i, id in enumerate(list_id):
        temp = [0,0,0,0,0,0,0,0,0,0]
        temp[int(id[1])-1] = 1
        Y.append(temp)

        X.append(make_input_features(data,cast_list,feature_dict,i))

    return X, Y


def make_input_features(data, cast_list, feature_dict, i):

    x = []

    for key, value in data.items():
        if key == "cast":
            for citem in cast_list:
                if citem in value[i]:
                    x.append(1)
                else:
                    x.append(0)

        elif not isinstance(value[0], int) and not isinstance(value[0], float):
            for citem in feature_dict[key]:
                if citem in value[i]:

                    x.append(1)

                else:
                    x.append(0)
        else:
            x.append(value[i])

    return x

def read_excel(data_file):
    i=0
    list_id = []
    with open(data_file) as input:
        csv_reader = csv.reader(input, delimiter=',')
        for row in csv_reader:
            if i != 0:
                list_id.append([row[1][2:],row[15]])
            i+= 1
    return list_id

def query_imdb(id,features,dict):

    ia = IMDb()
    movie = ia.get_movie(id)
    #print(movie.infoset2keys)

    try:
        for feature in features:

            try:
                first = movie[feature][0]['name']
                templist = []
                for index in movie[feature]:
                    templist.append(index['name'])
                dict[feature].append(templist)
            except Exception as e:
                dict[feature].append(movie[feature])
    except Exception as e:
        print(e)


    return dict

def vector_features(info, feature, n_actors):

    unique_values = {}
    all_features = []

    for key, value in info.items():
        if key == 'cast':
            for feature_list in value:
                for feature in feature_list:
                    all_features.append(feature)

        elif not isinstance(value[0], int) and not isinstance(value[0], float):
            unique_values[key] = []
            for feature_list in value:
                for feature in feature_list:
                    if feature not in unique_values[key]:
                        unique_values[key].append(feature)

    # cast_dict = {i: all_features.count(i) for i in all_features}
    #
    # cast_dict = {k: v for k, v in sorted(cast_dict.items(), key=lambda item: item[1], reverse=True)}

    cast_dict = {k: v for k, v in sorted({i: all_features.count(i) for i in all_features}.items(), key=lambda item: item[1], reverse=True)}
    cast_dict = take(n_actors, cast_dict.items())
    cast_list = []
    for item in cast_dict:
        cast_list.append(item[0])

    return cast_list, unique_values

def take(n, iterable):
    return list(islice(iterable, n))

def title_to_id(title):
    data_URL = 'http://www.omdbapi.com/?apikey=' + 'c24d8283'
    params = {
        't': title
    }
    response = requests.get(data_URL, params=params).json()
    id = response['imdbID'][2:]
    return id
