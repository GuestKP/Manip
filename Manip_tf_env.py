import pygame
import _thread
from random import shuffle
from time import sleep


class Manipulator:
    K = 10
    COLORS = [None, 'red', 'orange', 'yellow', 'green', 'blue', 'purple', 'brown', 'magenta']

    def __init__(self, n_boxes):
        tmp = 512/((n_boxes+1)*6+1)
        self.SIZES = {'gap': tmp, 'sum': 6*tmp, 'box': 5*tmp}
        self.boxes = [i for i in range(n_boxes+1)]
        self.n_boxes = n_boxes
        self.frame = 0
        self.window = None
        self.need_render = -1
        self.thread2 = _thread.start_new_thread(self.tread_main, ())
        # 1. почему нужет thread:
        #    если Manipulator.render вызвано ОДНАЖДЫ, то без thread окошко зависает.
        # 2. почему в Manipulator.render находится цикл с sleep(0):
        #    создание окошка зависает при любых командах (я хз почему),
        #    кроме sleep() и print()
        self.reset()

    def reset(self):
        shuffle(self.boxes)
        self.frame = 0
        return self.boxes

    def count_sum(self):
        return sum([(0 if idx+1 == box else -self.K) for idx, box in enumerate(self.boxes[:-1])]) - self.frame

    def step(self, pos):
        self.frame += 1
        if self.boxes[pos[0]] == 0 or self.boxes[pos[1]] != 0:  # if nothing in "From" or smth in "To"
            s = self.count_sum() - self.K*self.n_boxes  # penalty for error move
        else:
            self.boxes[pos[1]] = self.boxes[pos[0]]
            self.boxes[pos[0]] = 0  # move box
            s = self.count_sum()
        return self.boxes, s

    def tread_main(self):
        pygame.init()
        while True:
            if self.need_render >= 0:
                if self.window is None:
                    self.window = pygame.display.set_mode((512, 528))
                    pygame.display.set_caption("Labyrinth")

                self.window.fill(pygame.Color("black"))
                for pos, box in enumerate(self.boxes):
                    if box != 0:
                        pygame.draw.rect(self.window, pygame.Color(self.COLORS[box]),
                                         (self.SIZES['gap']+pos*self.SIZES['sum'],
                                          512-self.SIZES['sum'], self.SIZES['box'], self.SIZES['box']))
                    if pos != self.n_boxes:
                        pygame.draw.line(self.window, pygame.Color(self.COLORS[pos+1]),
                                         (self.SIZES['gap']+pos*self.SIZES['sum'], 520),
                                         ((pos+1)*self.SIZES['sum'], 520), 5)
                pygame.display.update()
                pygame.display.flip()

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        self.window = None
                        self.need_render = False
                self.need_render = 0

    def render(self):
        if self.need_render == -1:
            self.need_render = 1
            while self.need_render == 1:
                sleep(0)
