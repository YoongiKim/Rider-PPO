from RiderEnvironment.environment import RiderEnv
from model import AE_Model
import cv2

env = RiderEnv()
env.reset()
ae = AE_Model(100, 100)

while True:
    obs, reward, done, score = env.step([0])
    if done:
        env.reset()

    cv2.imshow('AE', ae.fit(obs, 100, 100))
    cv2.moveWindow('AE', 300, 500)