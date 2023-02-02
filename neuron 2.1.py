import neunet as nn
import numpy as np

n = nn.NeuralNetwork([784, 18, 18, 10], ['identity', 'identity', 'identity', 'sigmoid'], 0, 0.01, 1)
nn.load_weights(n)

import pygame as pg

sc = pg.display.set_mode((280, 280))
clock = pg.time.Clock()

coords = np.array([[0,0]])
coords = coords[:-1]
input_data = np.zeros([28, 28])
fps = 120
run = True
while run:
    for i in pg.event.get():
        if i.type == pg.QUIT:
            run = False
        if i.type == pg.MOUSEBUTTONDOWN:
            if i.button == 3:
                coords -= 1
                for j in coords:
                    input_data[j[0], j[1]] = 1

                input_data = np.transpose(input_data)
                input_data = [np.array([[input_data]]).flatten()]
                print(np.argmax(n.check(input_data)))

                coords = np.array([[0, 0]])
                coords = coords[:-1]
                input_data = np.zeros([28, 28])
                sc.fill((0, 0, 0))
                c = 0
                for i in range(28):
                    c += 10
                    pg.draw.aaline(sc, (100, 100, 100), [0, c], [280, c])
                    pg.draw.aaline(sc, (100, 100, 100), [c, 0], [c, 280])

    pressed = pg.mouse.get_pressed()
    pos = pg.mouse.get_pos()
    if pressed[0]:
        if pos[0] <= 270 and pos[1] <= 270:
            coord = np.array([int(np.floor(pos[0] / 10)), int(np.floor(pos[1] / 10))]) + 1
            coords = np.append(coords, np.array([coord]), axis=0)
            coords = np.unique(coords, axis=0)
            for i in coords:
                pg.draw.rect(sc, (255, 255, 255), [i * 10 - 10, [10, 10]])

    c = 0
    for i in range(28):
        c += 10
        pg.draw.aaline(sc, (100, 100, 100), [0, c], [280, c])
        pg.draw.aaline(sc, (100, 100, 100), [c, 0], [c, 280])


    pg.display.update()
    clock.tick(fps)
