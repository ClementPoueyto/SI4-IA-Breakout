# Based on the excellent
# https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
# And on lab / session 5 IA
# Girard Alexandre & Poueyto Clement

import os
import gym
import cv2
import argparse
import sys
import glob
import numpy as np
import pickle
from keras import backend as K
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense, Reshape
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam, Adamax, RMSprop
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Dropout, Flatten
from keras.layers.convolutional import UpSampling2D, Conv2D

# Parametres fixes
input_dim = 80 * 80
gamma = 0.99
update_frequency = 1
learning_rate = 0.1
resume = False
render = True
# Initialize
env = gym.make("Breakout-v0")
number_of_inputs = 3 # do nothing, right, left
#number_of_inputs = 4
observation = env.reset()
prev_screen = True
xs, dlogps, drs, probs = [], [], [], []
running_reward = True
reward_sum = 0
episode_number = 0
train_X = []
train_y = []

# Environment settings
EPISODES = 150

# Exploration settings
epsilon = 1  # starting epsolon
#epsilon = 0.1  # starting epsolon  
#EPSILON_DECAY = 0.998
MIN_EPSILON = 0.01
EPSILON_DECAY = (epsilon - MIN_EPSILON) /EPISODES


def breakout_preprocess_screen(I):
    #plt.imshow(I)
    #plt.show()
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    #plt.imshow(I, cmap='gray')
    #plt.show()
    return I.astype(np.float).ravel()

def preprocess(screen, width, height, targetWidth, targetHeight):
	#plt.imshow(screen)
	#plt.show()
	screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
	screen = screen[20:300, 0:200]  # crop off score
	screen = cv2.resize(screen, (targetWidth, targetHeight))
	screen = screen.reshape(targetWidth, targetHeight) / 255
	#plt.imshow(np.array(np.squeeze(screen)), cmap='gray')
	#plt.show()
	return screen

def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0:
            running_add = 0
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def learning_model(input_dim=80*80, model_type=1):
    model = Sequential()
    if model_type == 0:
        model.add(Reshape((1, 80, 80), input_shape=(input_dim,)))
        model.add(Flatten())
        model.add(Dense(200, activation='relu'))
        model.add(Dense(number_of_inputs, activation='softmax'))
        opt = RMSprop(lr=learning_rate)
    else:
        model.add(Reshape((1, 80, 80), input_shape=(input_dim,)))
        model.add(Conv2D(32, 9, 4, 
                  padding='same', activation='relu'))
        model.add(Flatten())
        model.add(Dense(16, activation='relu'))
        model.add(Dense(number_of_inputs, activation='softmax'))
        opt = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=opt)
    if resume == True:
        model.load_weights('breakout_model_checkpoint.h5')
    return model


model = learning_model()
# Begin training
for episode in range(0,EPISODES):
    observation = env.reset()
    currentLives = 5  # starting lives for episode
    done = False
    env.step(1)
    while not done:
        if render:
            env.render()
        # soustraction d'image
        cur_screen = breakout_preprocess_screen(observation)
        x = cur_screen - prev_screen if prev_screen is not None else np.zeros(input_dim)
        prev_screen = cur_screen
        # Predict probabilities from the Keras model
        aprob = ((model.predict(x.reshape([1, x.shape[0]]), batch_size=1).flatten()))
        aprob = aprob/np.sum(aprob)
        # Sample action
        #action = np.random.choice(number_of_inputs, 1, p=aprob)
        # Append features and labels for the episode-batch
        xs.append(x)
        probs.append(
            (model.predict(x.reshape([1, x.shape[0]]), batch_size=1).flatten()))
        aprob = aprob/np.sum(aprob)
        prob_aleat = np.random.random()
        # Get action from learning model
        if prob_aleat > epsilon:
            #action = np.random.choice(number_of_inputs, 1, p=aprob)[0]
            action = np.argmax(aprob)
            print(action)
            if action == 1 or action == 2 :
                action += 1
            #action = 2 if np.random.uniform() < aprob else 3  # roll the dice!
		# Get random action
        else:
            action = np.random.randint(3)
            if action == 1 or action == 2 :
                action += 1
        
        observation, reward, done, info = env.step(action)

        if action == 2 or action == 3 :
            action -= 1
        y = np.zeros([number_of_inputs])
        y[action] = 1
        dlogps.append(np.array(y).astype('float32') - aprob)

        if info["ale.lives"] < currentLives:
            currentLives = info["ale.lives"]
            reward = 0
            env.step(1)
        reward_sum += reward
        if render:
            env.render()

        drs.append(reward)
        if done:
            episode_number += 1
            epx = np.vstack(xs)
            epdlogp = np.vstack(dlogps)
            epr = np.vstack(drs)
            discounted_epr = discount_rewards(epr)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)
            epdlogp *= discounted_epr
            # Slowly prepare the training batch
            train_X.append(xs)
            train_y.append(epdlogp)
            xs, dlogps, drs = [], [], []
            # Periodically update the model
            if episode_number % update_frequency == 0:
                y_train = probs + learning_rate * \
                    np.squeeze(np.vstack(train_y))  # Hacky WIP
                #y_train[y_train<0] = 0
                #y_train[y_train>1] = 1
                #y_train = y_train / np.sum(np.abs(y_train), axis=1, keepdims=True)
                #print ("Training Snapshot:")
                #print (y_train)
                model.train_on_batch(np.squeeze(np.vstack(train_X)), y_train)
                # Clear the batch
                train_X = []
                train_y = []
                probs = []
                # Save a checkpoint of the model
                os.remove('breakout_model_checkpoint.h5') if os.path.exists(
                    'breakout_model_checkpoint.h5') else None
                model.save_weights('breakout_model_checkpoint.h5')
            # Reset the current environment nad print the current results
            running_reward = reward_sum if running_reward is None else running_reward * \
                0.99 + reward_sum * 0.01
            print ("Environment reset imminent. Total Episode Reward: %f. Running Mean: %f" % (reward_sum, running_reward))
            reward_sum = 0
            observation = env.reset()
            prev_screen = True
            print("Epsilon : "+str(epsilon))
            # Decay Epsilon
            if epsilon > MIN_EPSILON:
                epsilon -= EPSILON_DECAY
                epsilon = max(MIN_EPSILON, epsilon)
        if reward != 0:
            print(("Episode "+str(episode_number)+" Result: "+str(reward)))