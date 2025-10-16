import numpy as np
import tensorflow as tf
from collections import deque
import random
import os

def huber_loss(y_true, y_pred):
    return tf.keras.losses.Huber()(y_true, y_pred)

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.0001, gamma=0.95,
                 epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 batch_size=32, memory_capacity=10000, target_update_freq=100):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_capacity) # experience replay 
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.step_counter = 0
        self.model = self._build_model(learning_rate)
        self.target_model = self._build_model(learning_rate)
        self.target_model.set_weights(self.model.get_weights())
        self.optimizer = tf.keras.optimizers.Adam(learning_rate, clipnorm=1.0) # clipnorm=1 for consistent training

    def _build_model(self, learning_rate):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.state_size,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss=huber_loss, optimizer=tf.keras.optimizers.Adam(learning_rate))
        return model

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) #save in experience replay for learn from past

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice([0, 1])
        q_values = self.model.predict(state[np.newaxis, :], verbose=0) #two dimmentional array without extra message for batch
        return np.argmax(q_values[0]) #greedy choice

    def choose_action_with_q(self, state): #same as past but with q_values(for save in csv files)
        q_values = self.model.predict(state[np.newaxis, :], verbose=0)[0]
        if np.random.rand() <= self.epsilon:
            action = np.random.choice([0, 1])
            return action, q_values
        return np.argmax(q_values), q_values

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        #split experience section for experience replay
        indices = np.random.choice(len(self.memory), size=self.batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        states = np.array(states)
        next_states = np.array(next_states)
        #  Q_values...  and double dqn   for not overestimating
        current_q = self.model.predict(states, verbose=0)
        next_actions = np.argmax(self.model.predict(next_states, verbose=0), axis=1)
        next_q = self.target_model.predict(next_states, verbose=0)
        targets = current_q.copy()
        for i in range(self.batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * next_q[i][next_actions[i]] #bellman 
        with tf.GradientTape() as tape:  # active the gradian tape record the gradian of loss according to weights then update with adam
            q_values = self.model(states)
            loss = huber_loss(targets, q_values)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables)) # mini batch gradient
        self.step_counter += 1
        if self.step_counter % self.target_update_freq == 0:
            self.target_model.set_weights(self.model.get_weights())
        

    def save_model(self, path):
        self.model.save(path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path, custom_objects={'huber_loss': huber_loss}) # coustom object correct the error huber.getconfig
        self.target_model = tf.keras.models.load_model(path, custom_objects={'huber_loss': huber_loss})
        print(f"Model loaded from {path}")