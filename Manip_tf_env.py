import pygame
import _thread
from time import sleep
import numpy as np


class Manipulator:
    K = 4
    COLORS = list(
        set([l.translate({ord(i): None for i in '0123456789'}) for l in list(pygame.colordict.THECOLORS.keys())]))

    def __init__(self, n_boxes):
        tmp = 512/((n_boxes)*6+1)
        self.SIZES = {'gap': tmp, 'sum': 6*tmp, 'box': 5*tmp}
        self.boxes = np.array(list(range(n_boxes)))
        self.need = self.boxes.copy()
        self.n_boxes = n_boxes
        self.window = None
        self.need_render = -1
        # self.thread2 = _thread.start_new_thread(self.tread_main, ())
        # 1. почему нужет thread:
        #    если Manipulator.render вызвано ОДНАЖДЫ, то без thread окошко зависает.
        # 2. почему в Manipulator.render находится цикл с sleep(0):
        #    создание окошка зависает при любых командах (я хз почему),
        #    кроме sleep() и print()
        # self.reset()

    def reset(self):
        np.random.shuffle(self.boxes)
        return self.boxes

    def count_sum(self):
        return ((self.boxes == self.need)-1).sum()

    def step(self, fr, to):
        self.boxes[to], self.boxes[fr] = self.boxes[fr], self.boxes[to]
        s = self.count_sum()
        return self.boxes, s, s == 0

    def tread_main(self):
        pygame.init()
        while True:
            if self.need_render == 0:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        self.window = None
                        self.need_render = False

            if self.need_render == 1:
                if self.window is None:
                    self.window = pygame.display.set_mode((512, 528))
                    pygame.display.set_caption("Labyrinth")

                self.window.fill(pygame.Color("black"))
                for pos, box in enumerate(self.boxes):
                    pygame.draw.rect(self.window, pygame.Color(self.COLORS[box]),
                                     (self.SIZES['gap']+pos*self.SIZES['sum'],
                                      512-self.SIZES['sum'], self.SIZES['box'], self.SIZES['box']))
                    pygame.draw.line(self.window, pygame.Color(self.COLORS[pos]),
                                     (self.SIZES['gap']+pos*self.SIZES['sum'], 520),
                                     ((pos+1)*self.SIZES['sum'], 520), 5)
                pygame.display.update()
                pygame.display.flip()

                self.need_render = 0

    def render(self):
        self.need_render = 1
        while self.need_render == 1:
            sleep(0)
