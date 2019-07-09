import gym

import random
import numpy as np
import keras
from keras import backend
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

def createModel():
    # Sequential() creates the foundation of the layers.
    model = Sequential()
    # 'Dense' is the basic form of a neural network layer
    # Input Layer of state size(4) and Hidden Layer with 24 nodes
    model.add(Dense(24, input_dim=4, activation='relu'))
    # Hidden layer with 24 nodes
    model.add(Dense(24, activation='relu'))
    # Output Layer with # of actions: 2 nodes (left, right)
    model.add(Dense(1, activation='linear'))
    # print model summary
    model.summary()
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

    return model

def trainModel(model, data, checkpoint):
    X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size=.25)

    model.fit(X_train, y_train, validation_data=[X_test, y_test], verbose=1,
                callbacks=[checkpoint], epochs=20, batch_size=10000,shuffle=True)

    return model

def genTrainingData(env, min_score, train_episodes_n, train_obs_n=500):
    X, Y = [], []
    for i_episode in range(train_episodes_n):
        prev_state = env.reset()
        X_tmp, Y_tmp = [], []
        score = 0

        for _ in range(train_obs_n):
            action = random.randrange(0, 2)
            new_state, reward, done, _ = env.step(action)

            X_tmp.append(prev_state)
            Y_tmp.append(action)

            score += reward
            prev_state = new_state
    
            if done:
                if score > min_score:
                    X += X_tmp
                    Y += Y_tmp
                    # print('Episode: {} - score: {}'.format(i_episode, score))
                break

    X = np.array(X)
    Y = np.array(Y)

    np.savez('data/train', X=X, Y=Y)

    print('Shape of X:', X.shape)
    print('Shape of target:', Y.shape)

def play(model):
    episodes_n = 10
    min_score = 1000000
    max_score = -1
    avg_score = 0

    for _ in range(episodes_n):
        score = 0
        prev_state = env.reset()

        while (True):
            env.render()
            state = np.reshape(np.array(prev_state), (1, 4))
            pred = model.predict(state)
            action = 0 if pred[0][0] < .5 else 1

            new_state, reward, done, _ = env.step(action)

            score += reward 
            prev_state = new_state
            if done:
                if score < min_score:
                    min_score = score
                if score > max_score:
                    max_score = score
                avg_score += score
                print('Game ended with score:', score)
                break

    print('AVG score:', avg_score / episodes_n)
    print('MAX score:', max_score)
    print('MIN score:', min_score)


if __name__ == '__main__':
    EPISODES = 50000
    MIN_TRAIN_SCORE = 50

    env = gym.make('CartPole-v0')
    checkpoint = ModelCheckpoint('model/model_dnn.h5', monitor='val_loss', verbose=1, save_best_only=True)
    # genTrainingData(env, MIN_TRAIN_SCORE, EPISODES)

    dat = np.load('data/train.npz')
    train_data = dat['X'], dat['Y']

    model = createModel()
    trained_model = trainModel(model, train_data, checkpoint)
    play(trained_model)
