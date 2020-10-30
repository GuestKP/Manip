from Manip_tf_env import Manipulator
# from new import Storage
from time import sleep, time
from random import randint

import gym
import tensorflow as tf
import tf_slim as slim
from tensorflow.python.saved_model import tag_constants
import numpy as np


try:
    xrange = xrange
except:
    xrange = range

gamma = 0.5  # коэффициент дисконтирования

def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class Agent():
    def __init__(self, lr, s_size, a_size, h_size):
        # Ниже инициализирована feed-forward часть нейросети.
        # Агент оценивает состояние среды и совершает действие
        self.state_in = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
        hidden = slim.fully_connected(self.state_in, h_size,
                                      biases_initializer=None, activation_fn=tf.nn.relu)
        hidden_2 = slim.fully_connected(hidden, h_size,
                                       biases_initializer=None, activation_fn=tf.nn.relu)
        self.output = slim.fully_connected(hidden_2, a_size,
                                           activation_fn=tf.nn.softmax, biases_initializer=None)
        self.chosen_action = tf.argmax(self.output, 1)  # выбор действия

        # Следующие 6 строк устанавливают процедуру обучения.
        # Нейросеть принимает на вход выбранное действие
        # и соответствующий выигрыш,
        # чтобы оценить функцию потерь и обновить веса модели.
        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)

        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder

        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]),
                                             self.indexes)
        # функция потерь
        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs) *
                                    self.reward_holder)

        tvars = tf.trainable_variables()
        self.exported = tf.trainable_variables()
        self.gradient_holders = []
        for idx, var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
            self.gradient_holders.append(placeholder)

        self.gradients = tf.gradients(self.loss, tvars)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders,
                                                          tvars))


tf.reset_default_graph()  # Очищаем граф tensorflow

boxes = 4
actions = [[i, j] for i in range(boxes) for j in range(boxes) if i != j]
myAgent = Agent(lr=0.5, s_size=boxes, a_size=boxes*(boxes-1), h_size=32)  # Инициализируем агента
# myAgent = Agent(lr=1e-2, s_size=4, a_size=2, h_size=8)  # Инициализируем агента
saver = tf.train.Saver()

# env = Storage(32, 32, ex='storage.txt')
env = Manipulator(boxes)
# env = gym.make('CartPole-v0')
choice = list(range(boxes))

total_episodes = 5000  # Количество итераций обучения
max_ep = 200
update_frequency = 50
winned_eps = 0
init = tf.global_variables_initializer()

print('Point 1')

with tf.Session() as sess:
    sess.run(init)
    total_reward = []
    gradBuffer = sess.run(tf.trainable_variables())
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

    for i in range(total_episodes):
        st = env.reset()
        sum_rew = 0
        ep_history = []
        for j in range(max_ep):
            a_dist = sess.run(myAgent.output, feed_dict={myAgent.state_in: [st]})[0]
            act = a_dist.argmax()
            # act = np.random.choice(range(len(a_dist)), p=a_dist)
            st1, rew, done = env.step(actions[act][0], actions[act][1])  # Получить награду за совершенное действие
            # st1, rew, done, _ = env.step(act)
            ep_history.append([st, act, rew, st1])
            st = st1
            sum_rew += rew
            if done:
                winned_eps += 1
                ep_history = np.array(ep_history)
                ep_history[:, 2] = discount_rewards(ep_history[:, 2])
                feed_dict = {myAgent.reward_holder: ep_history[:, 2],
                             myAgent.action_holder: ep_history[:, 1],
                             myAgent.state_in: np.vstack(ep_history[:, 0])}
                grads = sess.run(myAgent.gradients, feed_dict=feed_dict)
                for idx, grad in enumerate(grads):
                    gradBuffer[idx] += grad

                if i % update_frequency == 0 and i != 0:
                    feed_dict = dictionary = dict(zip(myAgent.gradient_holders,
                                                      gradBuffer))
                    _ = sess.run(myAgent.update_batch, feed_dict=feed_dict)
                    for ix, grad in enumerate(gradBuffer):
                        gradBuffer[ix] = grad * 0

                total_reward.append(sum_rew)
                break

        if j == max_ep-1:
            total_reward.append(sum_rew - 50)

        if i % 100 == 0 and i != 0:
            print(np.mean(total_reward[-100:]), '\t--- Progress: ', ((i) * 100 / total_episodes),
                  '% --- Done param: ', len(total_reward), ' from ', total_episodes, 'episodes (', winned_eps, ' winned)')
            winned_eps = 0
    pass
