import pygame
import pygame.freetype
import numpy as np
import functools

screen_width = 600
cell_space = 5
cell_res_w = 5
cell_res_h = 5
cell_h = cell_w = int((275 - (cell_res_w + 1) * cell_space) /
                      cell_res_w)  # calculate cell size

screen_height = (cell_res_h + 1) * cell_space + \
    cell_res_h * cell_h  # calculate window size

pygame.init()
screen = pygame.display.set_mode([screen_width, screen_height])
pygame.display.set_caption('Perceptron GUI')  # set window title

# contains logical representation of states of cells
squares = np.zeros((cell_res_h, cell_res_w))

rectangles = []  # contains interface rectangles


def button_digit_click(digit):
    print(digit)


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


buttons = []


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
    buttons.append(
        Button('pred', (left + 55, top), (0, 162, 255), functools.partial(predict)))
    buttons.append(
        Button('noise', (left+2*55, top), (135, 132, 132), functools.partial(noise_input)))


def draw():
    left = top = cell_space
    rows, cols = squares.shape
    # draw squares
    for row in range(rows):
        for col in range(cols):
            rectangles[row][col] = pygame.Rect(left, top, cell_w, cell_h)
            if squares[row, col] == 0:
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
                            squares[row][col] = not squares[row][col]
                # check buttons for click
                for i in range(len(buttons)):
                    if (buttons[i].rect.collidepoint(mouse_position)):
                        buttons[i].click()
                draw()
    pygame.quit()


def digit_click(number):
    print('clicked ', number)


def train():
    print('train')


def predict():
    print('predict')


def noise_input():
    print('noise')


def noisy(data, noise_prob=0.1):
    noise = np.random.binomial(1, noise_prob, len(np.ravel(data)))
    data = np.ravel(data)
    for i in range(len(data)):
        if noise[i] == 1:
            data[i] = 1 if data[i] == -1 else - 1

    return data.reshape(squares.shape)


if __name__ == "__main__":
    main()
