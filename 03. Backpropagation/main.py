import pygame
import numpy as np
import random
import matplotlib.pyplot as plt
from example import Point
from example import Example
from neural_network import NeuralNetwork

arm_length = 100.0
x_translation = 318
y_translation = 230

pygame.init()
pygame.font.init()
screen = pygame.display.set_mode([600, 500])
ex = Example(arm_length)
e = ex.generate(10000)


def find_joints(angles):
    alpha = np.pi - angles[0]
    beta = angles[1]*(-1.0)
    first_joint = translate(Point(x_translation, y_translation), alpha)
    second_joint = translate(first_joint, np.pi - beta + alpha)
    return first_joint, second_joint


def translate(center, angle):
    return Point(center.x + arm_length * np.sin(angle), center.y - arm_length * np.cos(angle))


def unstandarize(angles):
    return np.array(angles)*np.pi


def draw_range():
    for _ in e[0]:
        pygame.draw.circle(screen, (128, 0, 0),
                           (_[0]+x_translation, _[1]+y_translation), 1.1)


def main():

    x_train = (np.array(e[0]) + arm_length * 2) / (arm_length * 4) * 0.8 + 0.1
    y_train = np.array(e[1])/np.pi*0.8 + 0.1

    NN = NeuralNetwork()

    for i in range(10000):
        NN.train(x_train, y_train)

    err = NN.errors
    plt.plot(range(len(err)), err)
    plt.legend()
    plt.savefig('errors.pdf')

    screen.fill((255, 255, 255))
    image = pygame.image.load('robot.jpg')
    screen.blit(image, (0, 0))
    pygame.display.flip()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEMOTION:
                pos = pygame.mouse.get_pos()

                x = pos[0] - x_translation
                y = pos[1] - y_translation

                prediction = unstandarize(NN.forward(
                    ((x + arm_length*2)/(arm_length*4)*0.8 + 0.1, (-y + arm_length*2)/(arm_length*4)*0.8 + 0.1)))

                joints = find_joints(prediction)

                screen.fill((255, 255, 255))
                image = pygame.image.load('robot.jpg')
                screen.blit(image, (0, 0))
                pygame.draw.line(screen, (255, 0, 0), (x_translation, y_translation),
                                 (joints[0].x, joints[0].y), width=5)
                pygame.draw.line(
                    screen, (0, 0, 255), (joints[0].x, joints[0].y), (joints[1].x, joints[1].y), width=5)

                # draw_range()
                pygame.display.flip()
    pygame.quit()


if __name__ == "__main__":
    main()
