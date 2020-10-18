import pygame
import pygame.freetype
import numpy as np
import functools

screen_width = 550
cell_space = 5
cell_res_w = 5
cell_res_h = 7
cell_w = int((250 - (cell_res_w + 1) * cell_space) /
             cell_res_w)  # calculate cell size
cell_h = cell_w

screen_height = (cell_res_h + 1) * cell_space + \
    cell_res_h * cell_h+300  # calculate window size

pygame.init()
game_font = pygame.freetype.SysFont(pygame.font.get_default_font(), 15)
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
        game_font.render_to(
            screen, (self.pos[0]+5, self.pos[1]+5), self.text, (255, 255, 255))


buttons = []


def digit_click(number):
    print('clicked ', number)


def train():
    print('train')


def predict():
    print('predict')


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

    # add digit buttons
    for y in range(2):
        for x in range(5):
            digit = y*5+x
            buttons.append(
                Button(str(y*5+x), (270 + x * 55, (y + 1) * 5 + y * 30), (146, 145, 181), functools.partial(digit_click, y*5+x)))
    # add train and predict buttons
    buttons.append(
        Button('train', (270, buttons[len(buttons) - 1].pos[1] + 40), (255, 213, 0), functools.partial(train)))
    buttons.append(
        Button('pred', (270+55, buttons[len(buttons) - 2].pos[1] + 40), (0, 162, 255), functools.partial(predict)))


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


if __name__ == "__main__":
    main()
