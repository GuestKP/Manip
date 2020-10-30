import pygame
import _thread
from random import randint
from time import sleep
import numpy as np

COLORS = list(set([l.translate({ord(i): None for i in '0123456789'}) for l in list(pygame.colordict.THECOLORS.keys())]))
enum = enumerate

class Storage:
    K = 10

    def __init__(self, size_x, size_y, n_robots=1, ex=None):
        self.BOX_SIZE = min(512/size_x, 512/size_y)
        self.map = []
        self.size = {'x': size_x, 'y': size_y, 'n': n_robots}
        self.window = None
        self.need_render = -1
        if ex is None:
            self.map = [[(x%2)or(y%2) for x in range(self.size['x'])] for y in range(self.size['y'])]
        else:
            with open(ex, 'r') as f:
                self.map = np.array([[randint(0, 9) if x=='*' else -size_x-size_y-1
                                      for x in line.strip('\n')] for line in f])
        tmp = np.array(np.where(self.map == 0)).T
        self.robot = [size_x//2, size_y//2]
        self.reset()
        self.thread2 = _thread.start_new_thread(self.tread_main, ())
        # 1. почему нужет thread:
        #    если Manipulator.render вызвано ОДНАЖДЫ, то без thread окошко зависает.
        # 2. почему в Manipulator.render находится цикл с sleep(0):
        #    создание окошка зависает при любых командах (я хз почему),
        #    кроме sleep() и print()

    def get_left(self, x, y):
        return self.map[y][x-1] if x > 0 else 0

    def get_right(self, x, y):
        return self.map[y][x+1] if x < self.size['x']-1 else 0

    def get_up(self, x, y):
        return self.map[y-1][x] if y > 0 else 0

    def get_down(self, x, y):
        return self.map[y+1][x] if y < self.size['y']-1 else 0

    def get_state(self):
        self.state = [self.get_up(*self.robot), self.get_right(*self.robot),
                      self.get_down(*self.robot), self.get_left(*self.robot)]

    def reset(self):
        self.map = [[-1024 if x<0 else x for x in line] for line in self.map]
        target = [randint(0, self.size['x']-1), randint(0, self.size['y']-1)]
        while self.map[target[1]][target[0]] < 0:
            target = [randint(0, self.size['x']-1), randint(0, self.size['y']-1)]

        pos = [target]
        first = self.map[target[1]][target[0]]
        self.map[target[1]][target[0]] = 0
        for [x, y] in pos:
            if self.get_up(x, y) < self.map[y][x]:
                self.map[y-1][x] = self.map[y][x]-1
                pos.append([x, y-1])
            if self.get_right(x, y) < self.map[y][x]:
                self.map[y][x+1] = self.map[y][x]-1
                pos.append([x+1, y])
            if self.get_down(x, y) < self.map[y][x]:
                self.map[y+1][x] = self.map[y][x]-1
                pos.append([x, y+1])
            if self.get_left(x, y) < self.map[y][x]:
                self.map[y][x-1] = self.map[y][x]-1
                pos.append([x-1, y])
            if pos[-1] == self.robot:
                break
        self.map[target[1]][target[0]] = first

        self.get_state()
        return 0

    def step(self):
        tmp = [x if x<0 else -1024 for x in self.state]
        side = tmp.index(max(tmp))
        if side == 0:
                self.robot[1] -= 1
        elif side == 1:
                self.robot[0] += 1
        elif side == 2:
                self.robot[1] += 1
        elif side == 3:
                self.robot[0] -= 1

        self.get_state()

        return self.map[self.robot[1]][self.robot[0]] == -1

    def tread_main(self):
        pygame.init()
        while True:
            if self.need_render == 0:

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        self.window = None
                        self.need_render = False

            elif self.need_render == 1:
                if self.window is None:
                    self.window = pygame.display.set_mode((512, 512))
                    pygame.display.set_caption("Labyrinth")

                self.window.fill(pygame.Color("black"))
                for y in range(self.size['y']):
                    for x in range(self.size['x']):
                        if self.map[y][x] >= 0:
                            pygame.draw.rect(self.window, pygame.Color(COLORS[self.map[y][x]]),
                                             (self.BOX_SIZE*x, self.BOX_SIZE*y,
                                              self.BOX_SIZE-1, self.BOX_SIZE-1))
                pygame.draw.rect(self.window, pygame.Color('darkgreen'),
                                 (self.BOX_SIZE*self.robot[0]+2, self.BOX_SIZE*self.robot[1]+2,
                                  self.BOX_SIZE-5, self.BOX_SIZE-5))
                pygame.display.update()
                pygame.display.flip()
                self.need_render = 0

    def render(self):
        self.need_render = 1
        while self.need_render == 1:
            sleep(0)

env = Storage(32, 32, ex='storage.txt')
env.reset()
env.render()
while True:
    done = env.step()
    env.render()
    print(env.robot)
    if done:
        env.reset()
