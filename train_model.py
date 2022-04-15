from generate_xy import generate_xy_main
from load_model import load_model
import os
import shutil
from sklearn import svm
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import ast
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt



def main():
    #features = ['genres', 'cast', 'rating', 'year', 'director']
    features = ['genres', 'rating', 'year', 'director']

    loc = 'genre_rating_year_director'
    os.mkdir('data/'+loc)
    os.mkdir('saved_model/'+loc)
    n_actors = 500
    #data_file = 'films_test.csv'
    data_file = 'films_bram.csv'
    save_data = True


    X, y = generate_xy_main(features, n_actors, data_file)
    try:
        with open('data/'+loc +'/y.txt') as f:
            y = f.readlines()

        y = ast.literal_eval(y[0])

        with open('data/'+loc +'/x.txt') as f:
            X = f.readlines()

        X = ast.literal_eval(X[0])
    except FileNotFoundError:
        print('folder did not exist')

    if save_data:
        if os.path.exists('data/'+loc +'/x.txt'):
            os.remove('data/'+loc +'/x.txt')
        if os.path.exists('data/'+loc +'/y.txt'):
            os.remove('data/'+loc +'/y.txt')
        textfile_x = open('data/'+loc +'/x.txt', "w")
        textfile_y = open('data/'+loc +'/y.txt', "w")
        textfile_x.write(str(X))
        textfile_y.write(str(y))

    print('gathered all data')

    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.05)

    train_nn(X_train, X_test, y_train, y_test,save_data,loc)

    load_model()

def train_svm(X_train, X_test, y_train, y_test):
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train,y_train)

    print('Train Accuracy : %.3f' % clf.score(X_train, y_train))
    print('Test Accuracy : %.3f' % clf.score(X_test, y_test))

def train_nn(X_train, X_test, y_train, y_test,save_data,loc):

    model = keras.Sequential([
        keras.layers.Dense(units=len(X_train[0]), activation=tf.nn.relu, kernel_regularizer='l2'),
        keras.layers.Dense(units=500, activation=tf.nn.relu, kernel_regularizer='l2'),
        keras.layers.Dropout(.2),
        keras.layers.Dense(units=500, activation=tf.nn.relu, kernel_regularizer='l2'),
        keras.layers.Dense(units=10, activation=tf.nn.softmax)
    ])

    model.compile(
        optimizer='SGD',
        metrics=['accuracy'],
        loss=keras.losses.CategoricalCrossentropy()
    )
    print('Compiling, now training')

    model.fit(
        x=X_train,
        y=y_train,
        epochs = 20
    )

    print('Done training, now evaluating')
    # score = model.evaluate(
    #     X_test,
    #     y_test,
    #     )

    y_pred = model.predict(X_test)

    predicted_categories = tf.argmax(y_pred, axis=1)
    print(predicted_categories)
    true_categories = []
    for one_hot in y_test:
        true_categories.append(one_hot.index(1))
    true_categories = tf.convert_to_tensor(true_categories)
    print(true_categories)
    cm = confusion_matrix(predicted_categories, true_categories)
    print(cm)

    labels = [1,2,3,4,5,6,7,8,9,10]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    #print('Test_accuracy: ',  score[1])
    if save_data:
        shutil.rmtree("saved_model/"+loc)
        os.mkdir("saved_model/"+loc)
        model.save("saved_model/"+loc)
        with open("saved_model/"+loc+'/report.txt', 'w') as fh:
            # Pass the file handle in as a lambda function to make it callable
            model.summary(print_fn=lambda x: fh.write(x + '\n'))

if __name__ == '__main__':
    main()
