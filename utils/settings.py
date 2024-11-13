import pygame
pygame.init()
pygame.font.init()

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (200, 200, 200)
RED = (255, 0, 0)
BLUE = (0, 255, 0)
GREEN = (0, 0, 255)

FPS = 240

WIDTH, HEIGHT = 1100, 700

ROWS = COLS = 28

TOOLBAR_HEIGHT = HEIGHT - WIDTH

PIXEL_SIZE = 22

BG_COLOR = WHITE

DRAW_GRID_LINES = False

FONT = pygame.font.SysFont("Arial", 16, bold=True)


def get_font(size):
    return pygame.font.SysFont("comicsans", size)
