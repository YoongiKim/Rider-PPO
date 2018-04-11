"""
Copyright (c) <2018> YoongiKim

 See the file license.txt for copying permission.
"""

import numpy as np
import cv2
import time
import pyautogui
from RiderEnvironment.grabscreen import grab_screen
from RiderEnvironment import show_window
import threading
from RiderEnvironment import read_score
import gym

## PRESS CTRL + ALT + DEL to stop program

class PreviousFrameMixer:
    PreviousFrames = []

    def __init__(self, number_of_frames, height, width):
        self.height = height
        self.width = width
        self.len = number_of_frames

        self.clear()

    def clear(self):
        self.PreviousFrames = []
        for i in range(self.len):
            self.PreviousFrames.append(np.zeros(shape=(self.height, self.width), dtype=np.uint8))

    def stack_frame(self, img):
        self.PreviousFrames.append(img)
        self.PreviousFrames.pop(0)

    def get_mixed_frames(self): # mix previous frames by time to reduce memory
        result_img = np.zeros(shape=(self.height, self.width), dtype=np.uint8)

        for i in range(self.len):
            result_img = cv2.addWeighted(result_img, float(i/self.len), self.PreviousFrames[i], float(i+1/self.len), 0)

        return np.array(result_img)

class RiderEnv:
    LastScore = 0
    LastAction = 0
    capture_x = 8
    capture_y = 120
    capture_w = 296
    capture_h = 296
    obs_w = 100 # Must Change models.py 224 line, 287 line
    obs_h = 100
    step_count = 0
    same_score_count = 0

    frame_mixer = PreviousFrameMixer(4, obs_h, obs_w)

    def __init__(self):
        self.frame_mixer.clear()

        self.observation_space = \
            gym.spaces.Box(low=0, high=255,
                           shape=np.zeros(shape=(self.obs_h * self.obs_w), dtype=np.uint8).shape
                           , dtype=np.uint8)
        #self.action_space = gym.spaces.Box(low=0, high=1, shape=np.zeros(1).shape, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)

    def reset(self):
        print('env reset')
        show_window.ShowWindow()
        pyautogui.moveTo(155, 350)

        self.LastScore = 0
        self.LastAction = 0
        self.same_score_count = 0

        self.frame_mixer.clear()

        self.close_advertise_window()

        self.click()

        time.sleep(1.5)

        self.click()

        observation = np.zeros(shape=(self.obs_h, self.obs_w), dtype=np.uint8)

        return np.array(observation).flatten()

    def step(self, action):
        # observation, reward, done, score

        self.step_count += 1

        if float(action[0]) >= 0.5 and self.LastAction == 0:
            #print("mouse down")
            self.mouse_down()
            self.LastAction = 1

        elif float(action[0]) < 0.5 and self.LastAction == 1:
            #print("mouse up")
            self.mouse_up()
            self.LastAction = 0

        result_frame = self.get_frame()

        done = self.isDone(result_frame)
        main_menu = self.isMainMenu(result_frame)
        self.close_advertise_window()

        score = self.LastScore
        if self.step_count % 5 == 0:
            score = self.get_score(result_frame)
        # score = self.get_score(result_frame)
        if score <= self.LastScore:
            self.same_score_count += 1
            if self.same_score_count > 150:
                self.back_to_menu()

        else:
            self.same_score_count = 0

        reward = (score - self.LastScore) * 5 \
                 + 0.005*self.LastAction \
                 - self.same_score_count * 0.005

        self.LastScore = score

        if done:
            reward = score - self.LastScore
            #reward = -1*(100-self.LastScore)

        current_observation = self.__get_observation(result_frame)

        self.frame_mixer.stack_frame(current_observation)

        if self.step_count % 1 == 0:
            print("step: {}, reward: {}, done: {}, score: {}, action: {}"
                  .format(self.step_count, reward, done, score, action[0]))

        mixed_frame = self.frame_mixer.get_mixed_frames()
        self.show(mixed_frame, "obs", 313, 200)

        return mixed_frame.flatten(), reward, done, self.LastScore

    def close(self):
        cv2.destroyAllWindows()

    def __get_observation(self, screen):
        edge_screen = self.process_img(screen)
        return edge_screen

    def to_binary(self, img):
        retval, threshold = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
        return np.array(threshold)

    def render(self):
        return

    def get_frame(self):
        screen = grab_screen(region=(0, 0, 312, 578))
        return screen

    def process_img(self, image):
        # convert to gray
        processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # cut unused area
        y=self.capture_y
        x=self.capture_x
        h=self.capture_h
        w=self.capture_w
        processed_img = processed_img[y:y+h, x:x+w]

        processed_img = cv2.Canny(processed_img, threshold1=200, threshold2=300)
        #processed_img = cv2.GaussianBlur(processed_img, (5, 5), 0)
        processed_img = cv2.resize(processed_img, (self.obs_w, self.obs_h), cv2.INTER_AREA)

        return processed_img

    def get_score(self, image):
        x = 154-50
        y = 136-50
        w = 100
        h = 100

        score_image = image[y:y + h, x:x + w]

        score = read_score.read(score_image)

        if abs(score - self.LastScore) >= 10:
            score = self.LastScore

        return score

    def isDone(self, original_img):
        if self.mean_bgr(original_img[574, 10]) <= 4:
            return True
        else:
            return False

    def isMainMenu(self, original_img):
        if self.mean_bgr(original_img[475, 288]) >= 254 \
                and self.mean_bgr(original_img[466, 24]) >= 254:
            return True
        else:
            return False

    def mean_bgr(self, pixel):
        sum = 0
        for i in range(3):
            sum += pixel[i]
        sum /= 3
        return sum

    def close_advertise_window(self):
        frame = self.get_frame()
        done = self.isDone(frame)
        main_menu = self.isMainMenu(frame)

        while done and not main_menu:
            print('done: {}, main menu: {}'.format(done, main_menu))
            time.sleep(0.5)
            self.click(250, 163)
            self.click(260, 142)
            self.mouse_move_to_center()

            frame = self.get_frame()
            done = self.isDone(frame)
            main_menu = self.isMainMenu(frame)

    def back_to_menu(self):
        self.click(22, 60)
        time.sleep(1)
        self.click(153,353)
        time.sleep(1)

    def show(self, img, title, x=400, y=500):
        cv2.imshow(title, img)
        cv2.moveWindow(title, x, y)
        cv2.waitKey(1)

    def mouse_up(self):
        threading.Thread(target=pyautogui.mouseUp).start()
        # pyautogui.mouseUp()

    def mouse_down(self):
        self.mouse_move_to_center()
        # threading.Thread(target=pyautogui.mouseDown).start()
        pyautogui.mouseDown()

    def mouse_move_to_center(self):
        # threading.Thread(target=pyautogui.moveTo, args=[155, 350]).start()
        pyautogui.moveTo(155, 350)

    def click(self, x=155, y=350):
        # threading.Thread(target=pyautogui.click, args=[x, y]).start()
        pyautogui.click(x, y)
