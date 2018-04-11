"""
Copyright (c) <2018> YoongiKim

 See the file license.txt for copying permission.
"""

from RiderEnvironment import environment as rider_env
import random
import time

for i in range(3):
    env = rider_env.RiderEnv()
    done = False
    env.reset()

    while not done:
        rand = random.randrange(1, 2)
        obs, reward, done, score = env.step([rand])

env.close()
