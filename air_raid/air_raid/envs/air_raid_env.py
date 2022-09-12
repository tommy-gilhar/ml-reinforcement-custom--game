import copy

import numpy as np
import pygame
import time
import random
from pygame.surfarray import array3d
from pygame import display
import gym
from gym import spaces


# Sets up colors for the game using RGB Codes
BLACK = pygame.Color(0, 0, 0)
WHITE = pygame.Color(255, 255, 255)
RED = pygame.Color(255, 0, 0)
GREEN = pygame.Color(0, 255, 0)
check_errors = pygame.init()


class AirRaid(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        """
        Defines the initial game window size
        """
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(0, 255, shape=(500, 500, 3), dtype=np.uint8)
        self.frame_size_x = 200
        self.frame_size_y = 200
        self.game_window = pygame.display.set_mode(
            (self.frame_size_x, self.frame_size_y)
        )
        self.green_positions = []
        self.red_positions = []
        self.time_between_every_green = 7
        self.time_between_every_red = 25
        self.spawn_location = 30
        self.square_size = 20
        self.green_pos = self.spawn_square()
        self.red_pos = self.spawn_square()
        self.player_pos = [100, 170]
        self.action = 0
        self.score = 0
        self.steps = 0
        self.reset()
        self.STEP_LIMIT = 2000
        self.sleep = 0.1

    def reset(self, *args):
        """
        Resets the game, along with the default pac size and spawning food.
        """
        self.game_window.fill(BLACK)
        self.player_pos = [100, 170]
        self.green_pos = self.spawn_square()
        self.red_pos = self.spawn_square()
        self.action = 0
        self.score = 0
        self.steps = 0
        img = array3d(display.get_surface())
        img = np.swapaxes(img, 0, 1)
        return img

    def change_direction(self, player_pos):
        if player_pos[0] < 0:
            player_pos[0] += 20
        if player_pos[0] > self.frame_size_y - 20:
            player_pos[0] -= 20

        return player_pos

    def spawn_square(self):
        """
        Spawns green square in a random location on the window
        """
        return [
            random.randrange(1, (self.frame_size_x // 25)) * 25,
            self.spawn_location,
        ]

    def reward_calc(self, player, green_square, i):
        if self.collide(player, green_square):
            self.score += 1
            reward = 1
            self.green_positions[i][1] = self.frame_size_y + 50
        else:
            reward = 0
        return reward

    def draw_green(self, reward, player):
        # Create a new Green square every x steps
        if self.steps % self.time_between_every_green == 2:
            self.green_pos = self.spawn_square()
            self.green_positions.append(self.green_pos)
        # Check if there is any greens squares on the screen
        if self.green_positions:
            for i, pos in enumerate(self.green_positions):
                # Check if the square is out of bound and remove it, otherwise continue.
                if pos[1] >= self.frame_size_y:
                    self.green_positions[i] = None
                else:
                    green_square = pygame.draw.rect(
                        self.game_window, GREEN, pygame.Rect(pos[0], pos[1], 25, 25)
                    )

                    self.green_positions[i] = self.move_squares(pos)

                    reward += self.reward_calc(player, green_square, i)
            self.green_positions = list(filter(None, self.green_positions))

    def draw_red(self, reward, player):
        done = False
        if self.steps % self.time_between_every_red == 5:
            self.red_pos = self.spawn_square()
            if (self.green_pos[0] + 25) >= self.red_pos[0] >= (self.green_pos[0] - 25):
                self.red_pos = self.spawn_square()
            self.red_positions.append(self.red_pos)
        if self.red_positions:
            red_square = None

            for i, pos in enumerate(self.red_positions):
                if pos[1] == self.frame_size_y:
                    self.red_positions[i] = None
                else:
                    red_square = pygame.draw.rect(
                        self.game_window, RED, pygame.Rect(pos[0], pos[1], 25, 25)
                    )

                    self.red_positions[i] = self.move_squares(pos)
                reward, done = self.game_over(reward, player, red_square)
            self.red_positions = list(filter(None, self.red_positions))
        return done, reward

    def step(self, action):
        """
        what happens when your agent preforms the action on the env
        """
        reward = 0
        self.game_window.fill(BLACK)

        self.player_pos = self.move(action, self.player_pos)
        self.player_pos = self.change_direction(self.player_pos)

        player = pygame.draw.rect(
            self.game_window,
            WHITE,
            pygame.Rect(self.player_pos[0], self.player_pos[1], 20, 20),
        )
        # draw green
        self.draw_green(reward, player)

        # draw red
        done, reward = self.draw_red(reward, player)

        img = self.get_image_array_from_game()

        info = {"score": self.score}
        self.steps += 1
        time.sleep(self.sleep)
        self.display_score(WHITE, "times", 20)

        return img, reward, done, info

    def game_over(self, reward, player, red_square):
        """
        Checks if the player has touched the red square
        """
        # TOUCH Red
        collide = self.collide(player, red_square)
        if collide:
            self.score -= 1
            return -1, True

        if self.steps >= self.STEP_LIMIT:
            self.score -= 1
            return 0, True

        return reward, False

    def render(self, mode="human"):
        if mode == "human":
            display.update()

    def display_score(self, color, font, size):
        """
        Updates the score in top left
        """
        score_font = pygame.font.SysFont(font, size)
        score_surface = score_font.render("Score : " + str(self.score), True, color)
        score_rect = score_surface.get_rect()
        score_rect.midtop = (self.frame_size_x / 10, 15)
        self.game_window.blit(score_surface, score_rect)

    @staticmethod
    def get_image_array_from_game():
        img = array3d(display.get_surface())
        img = np.swapaxes(img, 0, 1)
        return img

    @staticmethod
    def move_squares(pos):
        pos[1] += 25
        return pos

    @staticmethod
    def move(action, player_pos):
        if action == 1:
            player_pos[0] -= 20
        if action == 0:
            player_pos[0] += 20
        return player_pos

    @staticmethod
    def collide(player, green_square):
        """
        Returns Boolean indicating if player has "hiten" the green square
        """
        collide = pygame.Rect.colliderect(player, green_square)
        return collide
