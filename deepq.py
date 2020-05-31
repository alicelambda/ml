import gym
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
import tensorflow as tf
import random
import time
import matplotlib.pyplot as plt
from collections import deque

env = gym.make("Breakout-ram-v0")
env = gym.wrappers.Monitor(env, "mowlorgebatch",
                           video_callable=lambda x: x % 20 == 0,
                           force=True)
INPUT_SHAPE = env.observation_space.shape
print(INPUT_SHAPE)
NUM_ACTIONS = env.action_space.n
print(NUM_ACTIONS)


def createRamModel():

    model = tf.keras.Sequential()
    model.add(layers.Dense(80, activation='relu', input_shape=INPUT_SHAPE))
    # output is the same size as number of outputs
    model.add(layers.Dense(60, activation='relu'))
    model.add(layers.Dense(NUM_ACTIONS, activation='linear'))

    model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                  loss='mse',       # mean squared error
                  metrics=['mae'])  # mean absolute error
    return model


target_model = createRamModel()
training_model = createRamModel()
training_model.set_weights(target_model.get_weights())

memory_actions = deque(maxlen=10000)
epsilon = 1
batch_size = 100
c = 300
episodes = []
for i in range(2000):
    total_reward = 0
    counter = 0
    obs = env.reset().reshape(1, -1)
    # the action the model takes is the output with the highest value
    action = np.argmax(target_model.predict(obs))
    done = False
    while not done:
        lastobs = obs
        obs, reward, done, info = env.step(action)
        total_reward += reward
        obs = obs.reshape(1, -1)
        if random.random() > epsilon:
            action = np.argmax(target_model.predict(obs))
        else:
            action = env.action_space.sample()
        step = [lastobs, action, reward, obs]
        memory_actions.append(step)
        if len(memory_actions) > 6000:
            print("training " + str(epsilon))
            # do training once we've sampled enough actions
            batch = np.asarray(random.sample(memory_actions, batch_size))
            current_states = np.concatenate([i[0] for i in batch])
            cur_q_vals = training_model.predict(current_states)
            next_states = np.concatenate([i[3] for i in batch])
            rewards = np.array([i[2] for i in batch])
            actions = to_categorical(
                np.array([i[1] for i in batch]), num_classes=NUM_ACTIONS)
            future_q_vals = training_model.predict(next_states)
            maxfuture_q = np.amax(future_q_vals, axis=1)
            updates = rewards + 0.99*maxfuture_q
            np.putmask(cur_q_vals, actions, updates.astype(
                'float32', casting='same_kind'))
            training_model.fit(current_states, cur_q_vals,
                               batch_size=batch_size)
            counter += 1
            epsilon = max(0.1,epsilon *0.999999)
            if counter > c:
                counter = 0
                print("SET WEIGHTS")
                target_model.set_weights(training_model.get_weights())

    episodes.append(total_reward)
