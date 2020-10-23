from Manip_tf_env import Manipulator
from time import time

env = Manipulator(8)
env.render()
state = env.reset()

a = True
t = time()
while True:
    if time()-t > 2:
        if state.index(0)+1 != len(state):
            act = [state.index(state.index(0)+1), state.index(0)]
        else:
            act = [[idx+1 != box for idx, box in enumerate(state)].index(True), len(state)-1]
        state_new, reward = env.step(act)
        print(state, act, state_new, reward)
        state = state_new
        t = time()
