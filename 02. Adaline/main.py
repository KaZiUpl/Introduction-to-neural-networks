import pygame
import pygame.freetype
import numpy as np
import functools
from data import number
from adaline import Adaline
import matplotlib.pyplot as plt
import random

screen_width = 600
cell_space = 5
cell_res_w = 10
cell_res_h = 10
cell_h = cell_w = int((275 - (cell_res_w + 1) * cell_space) /
                      cell_res_w)  # calculate cell size

screen_height = (cell_res_h + 1) * cell_space + \
    cell_res_h * cell_h  # calculate window size

pygame.init()
screen = pygame.display.set_mode([screen_width, screen_height])

pygame.display.set_caption('Adaline')


def digit_click(num):
    global squares
    squares = np.array(number[num])
    print(np.array(number[num]))
    draw()


def shift(direction):
    global squares
    if direction == 'up':
        squares = np.roll(squares, -1, axis=0)
    elif direction == 'down':
        squares = np.roll(squares, 1, axis=0)
    elif direction == 'left':
        squares = np.roll(squares, -1, axis=1)
    elif direction == 'right':
        squares = np.roll(squares, 1, axis=1)
    draw()


def clear():
    global squares
    squares = np.zeros((cell_res_h, cell_res_w))
    draw()


def train():
    for i in range(10):
        perceptrons[i].train(training_data, labels[i])
    print('trained')


def predict():
    print('predicted: ')
    pre = []
    for i in range(len(perceptrons)):
        print(i, perceptrons[i].output(np.ravel(squares)))
        pre.append(perceptrons[i].output(np.ravel(squares)))
    print('---')
    print(pre.index(max(pre)))
    print('=======')


def noise_input():
    global squares
    squares = noisyBer(squares, 0.07)
    draw()


def noisyBer(data, noise_prob=0.1):
    noise = np.random.binomial(1, noise_prob, len(np.ravel(data)))
    copy = np.ravel(np.copy(data))
    for i in range(len(copy)):
        if noise[i] == 1:
            copy[i] = 1 if copy[i] == 0 else 0

    return copy.reshape(data.shape)


def print_sq():
    global squares
    print('[')
    for row in range(len(squares)):
        row_str = '['
        for col in range(len(squares[row])):
            row_str += str(squares[row][col])
            if col < len(squares[row]) - 1:
                row_str += ','
        row_str += ']'
        if row < len(squares) - 1:
            row_str += ','
        print(row_str)
    print(']')


# contains logical representation of states of cells
squares = np.zeros((cell_res_h, cell_res_w))
rectangles = []  # contains interface rectangles
buttons = []  # contains interface buttons
perceptrons = []  # contains perceptrons

# create perceptrons
labels = []
for i in range(10):
    _perceptron = Adaline(i, cell_res_w * cell_res_h)
    perceptrons.append(_perceptron)

# prepare training data
training_data = [np.ravel(num) for num in number]
for i in range(10):
    _labels = [0 for _ in range(10)]
    _labels[i] = 1
    labels.append(_labels)


class Button:
    def __init__(self, text, pos, color,  function):
        self.text = text
        self.pos = pos
        self.color = color
        self.rect = pygame.Rect(pos[0], pos[1], 50, 30)
        self.function = function

    def click(self):
        self.function()

    def draw(self):
        pygame.draw.rect(screen, self.color, self.rect)
        font = pygame.font.SysFont('calibri', 20)
        text = font.render(self.text, 1, (255, 255, 255))
        x = self.pos[0] + 50/2 - text.get_width()/2
        y = self.pos[1] + 30/2 - text.get_height()/2
        screen.blit(text, (int(x), int(y)))


def init():
    left = top = cell_space
    rows, cols = squares.shape

    # init squares
    for row in range(rows):
        _rectangles = []
        for col in range(cols):
            rect = pygame.Rect(left, top, cell_w, cell_h)
            _rectangles.append(rect)
            left += cell_w + cell_space
        left = cell_space
        top += cell_h + cell_space
        rectangles.append(_rectangles)

    left = 300
    top = 5
    # add digit buttons
    for y in range(2):
        for x in range(5):
            digit = y*5+x
            buttons.append(
                Button(str(y * 5 + x), (left + x * 55, top), (146, 145, 181), functools.partial(digit_click, y * 5 + x)))
        top += 35
    # add train and predict buttons
    buttons.append(
        Button('train', (left, top), (255, 213, 0), functools.partial(train)))
    top += 35
    buttons.append(
        Button('clr', (left, top), (52, 52, 255), functools.partial(clear)))
    buttons.append(
        Button('pred', (left + 55, top), (0, 162, 255), functools.partial(predict)))
    buttons.append(
        Button('prt', (left + 2 * 55, top), (0, 162, 162), functools.partial(print_sq)))

    top += 35
    buttons.append(
        Button('^', (left + 55, top), (0, 179, 0), functools.partial(shift, 'up')))
    top += 35
    buttons.append(
        Button('<', (left, top), (0, 179, 0), functools.partial(shift, 'left')))
    buttons.append(
        Button('v', (left + 55, top), (0, 179, 0), functools.partial(shift, 'down')))
    buttons.append(
        Button('>', (left + 2 * 55, top), (0, 179, 0), functools.partial(shift, 'right')))


def draw():
    left = top = cell_space
    rows, cols = squares.shape
    # draw squares
    for row in range(rows):
        for col in range(cols):
            rectangles[row][col] = pygame.Rect(left, top, cell_w, cell_h)
            if squares[row][col] == 0:
                pygame.draw.rect(screen, (96, 96, 96), rectangles[row][col])
            else:
                pygame.draw.rect(screen, (146, 145, 181), rectangles[row][col])
            left += cell_w + cell_space
        left = cell_space
        top += cell_h + cell_space
    # draw buttons
    for i in range(len(buttons)):
        buttons[i].draw()

    pygame.display.flip()


def main():
    screen.fill((51, 51, 51))
    init()
    draw()
    pygame.display.flip()

    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_position = pygame.mouse.get_pos()
                # check squares for click
                for row in range(len(rectangles)):
                    for col in range(len(rectangles[row])):
                        if rectangles[row][col].collidepoint(mouse_position):
                            squares[row][col] = 1 if squares[row][col] == 0 else 0
                # check buttons for click
                for i in range(len(buttons)):
                    if (buttons[i].rect.collidepoint(mouse_position)):
                        buttons[i].click()
                draw()
    pygame.quit()


if __name__ == "__main__":
    main()
