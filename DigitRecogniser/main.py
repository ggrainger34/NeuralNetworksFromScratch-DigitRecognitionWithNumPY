from utils import *
from utils.bar import Bar

from network import Network
from network import Layer

from keras.datasets import mnist

import numpy as np

from utils.label import Label

WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Handwitten Digit Recogniser")

programIcon = pygame.image.load("HandwrittenDigit.png")
pygame.display.set_icon(programIcon)

def init_grid(rows, cols, color):
    grid = []

    for i in range(rows):
        grid.append([])
        for _ in range(cols):
            grid[i].append(color)

    return grid


def draw_grid(win, grid):
    for i, row in enumerate(grid):
        for j, pixel in enumerate(row):
            pygame.draw.rect(win, pixel, (j * PIXEL_SIZE, i *
                                          PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE))

    if DRAW_GRID_LINES:
        for i in range(ROWS + 1):
            pygame.draw.line(win, BLACK, (0, i * PIXEL_SIZE),
                             (WIDTH, i * PIXEL_SIZE))

        for i in range(COLS + 1):
            pygame.draw.line(win, BLACK, (i * PIXEL_SIZE, 0),
                             (i * PIXEL_SIZE, HEIGHT - TOOLBAR_HEIGHT))


def draw(win, grid, buttons, bars):
    win.fill(BG_COLOR)
    draw_grid(win, grid)

    for button in buttons:
        button.draw(win)

    for bar in bars:
        bar.draw(win)

    for label in labels:
        label.draw(win)

    pygame.display.update()


def get_row_col_from_pos(pos):
    x, y = pos
    row = y // PIXEL_SIZE
    col = x // PIXEL_SIZE

    if row >= ROWS:
        raise IndexError

    return row, col

def convertGridFormatToNeuralNetworkInput(grid):
    neuralNetworkGrid = np.zeros((28,28))

    for row in range(0,ROWS):
        for col in range(0,COLS):
            if grid[row][col] == BLACK:
                neuralNetworkGrid[row][col] = 0
            if grid[row][col] == GREY:
                neuralNetworkGrid[row][col] = 0.9
            if grid[row][col] == WHITE:
                neuralNetworkGrid[row][col] = 1

    return neuralNetworkGrid

(trainX, trainY), (testX, testY) = mnist.load_data()

#Initialise Network
network = Network()

#Define the learning rate
learningRate = 0.01

#Define Layers
a1 = Layer(784,10,False)
a2 = Layer(10,10,False)
a3 = Layer(10,10,False)
a4 = Layer(10,10,True)

#Add layers
network.addLayer(a1)
network.addLayer(a2)
network.addLayer(a3)
network.addLayer(a4)

#Load the saved weights and biases into the network
network.loadWeightsAndBiases()

run = True
clock = pygame.time.Clock()
grid = init_grid(ROWS, COLS, BLACK)
drawing_color = WHITE

button_y = 630
buttons = [
    Button(10, button_y, 100, 50, WHITE, "Draw", BLACK),
    Button(177, button_y, 100, 50, WHITE, "Erase", BLACK),
    Button(344, button_y, 100, 50, WHITE, "Clear", BLACK),
    Button(510, button_y, 100, 50, WHITE, "Predict", BLACK)
]

bar_values = [0] * 10
bars_x = 600
bars = [
    Bar(700, 10, 350, 50, RED, bar_values[0]),
    Bar(700, 70, 350, 50, RED, bar_values[1]),
    Bar(700, 130, 350, 50, RED, bar_values[2]),
    Bar(700, 190, 350, 50, RED, bar_values[3]),
    Bar(700, 250, 350, 50, RED, bar_values[4]),
    Bar(700, 310, 350, 50, RED, bar_values[5]),
    Bar(700, 370, 350, 50, RED, bar_values[6]),
    Bar(700, 430, 350, 50, RED, bar_values[7]),
    Bar(700, 490, 350, 50, RED, bar_values[8]),
    Bar(700, 550, 350, 50, RED, bar_values[9])
]

labels = [
    Label(640, 10, 50, 50, "0"),
    Label(640, 70, 50, 50, "1"),
    Label(640, 130, 50, 50, "2"),
    Label(640, 190, 50, 50, "3"),
    Label(640, 250, 50, 50, "4"),
    Label(640, 310, 50, 50, "5"),
    Label(640, 370, 50, 50, "6"),
    Label(640, 430, 50, 50, "7"),
    Label(640, 490, 50, 50, "8"),
    Label(640, 550, 50, 50, "9")
]

while run:
    clock.tick(FPS)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

        if pygame.mouse.get_pressed()[0]:
            pos = pygame.mouse.get_pos()

            try:
                row, col = get_row_col_from_pos(pos)
                grid[row][col] = drawing_color
                
                #Colour the area around the drawing colour grey
                if grid[row+1][col] != drawing_color:
                    grid[row+1][col] = GREY
                if grid[row][col+1] != drawing_color:
                    grid[row][col+1] = GREY
                if grid[row-1][col] != drawing_color:
                    grid[row-1][col] = GREY
                if grid[row][col-1] != drawing_color:
                    grid[row][col-1] = GREY

            except IndexError:
                for button in buttons:
                    if not button.clicked(pos):
                        continue

                    drawing_color = button.color
                    if button.text == "Clear":
                        grid = init_grid(ROWS, COLS, BLACK)
                        drawing_color = WHITE

                    if button.text == "Erase":
                        drawing_color = BLACK

                    if button.text == "Predict":
                        networkInput = convertGridFormatToNeuralNetworkInput(grid)

                        processedInput = np.transpose(np.array([networkInput.flatten()]))

                        bar_values = np.transpose(network.feedForward(processedInput))[0]

                        #Update bar size
                        bars = [
                            Bar(700, 10, 350, 50, RED, bar_values[0]),
                            Bar(700, 70, 350, 50, RED, bar_values[1]),
                            Bar(700, 130, 350, 50, RED, bar_values[2]),
                            Bar(700, 190, 350, 50, RED, bar_values[3]),
                            Bar(700, 250, 350, 50, RED, bar_values[4]),
                            Bar(700, 310, 350, 50, RED, bar_values[5]),
                            Bar(700, 370, 350, 50, RED, bar_values[6]),
                            Bar(700, 430, 350, 50, RED, bar_values[7]),
                            Bar(700, 490, 350, 50, RED, bar_values[8]),
                            Bar(700, 550, 350, 50, RED, bar_values[9])
                        ]

    draw(WIN, grid, buttons, bars)

pygame.quit()