import pygame
import numpy as np

screen_width = 300
cell_space = 5
cell_res_w = 3
cell_res_h = 7
cell_w = int((screen_width - (cell_res_w + 1) * cell_space) /
             cell_res_w)  # calculate cell size
cell_h = cell_w

screen_height = (cell_res_h + 1) * cell_space + \
    cell_res_h * cell_h + 200  # calculate window size

pygame.init()
screen = pygame.display.set_mode([screen_width, screen_height])
pygame.display.set_caption('Perceptron GUI')  # set window title


# contains logical representation of states of cells
squares = np.zeros((cell_res_w, cell_res_h))


rectangles = []  # contains interface rectangles


def init():
    left = top = cell_space
    rows, cols = squares.shape

    for row in range(rows):
        _rectangles = []
        for col in range(cols):
            rect = pygame.Rect(left, top, cell_w, cell_h)
            _rectangles.append(rect)
            left += cell_w + cell_space
        left = cell_space
        top += cell_h + cell_space
        rectangles.append(_rectangles)


def draw():
    left = top = cell_space
    rows, cols = squares.shape
    count = 0

    for row in range(rows):
        for col in range(cols):

            rectangles[row][col] = pygame.Rect(left, top, cell_w, cell_h)
            if squares[row, col] == 0:
                pygame.draw.rect(screen, (128, 128, 128), rectangles[row][col])
                count += 1
            else:
                pygame.draw.rect(screen, (0, 255, 157), rectangles[row][col])
                count += 1

            left += cell_w + cell_space
        left = cell_space
        top += cell_h + cell_space
    print(count)
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
                for row in range(len(rectangles)):
                    for col in range(len(rectangles[row])):
                        if rectangles[row][col].collidepoint(mouse_position):
                            squares[row][col] = not squares[row][col]
                draw()
    pygame.quit()


if __name__ == "__main__":
    main()
