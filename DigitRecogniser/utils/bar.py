from .settings import *

class Bar:
    def __init__(self, x, y, width, height, color, bar_value):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.bar_value = bar_value

    def draw(self, win):
        pygame.draw.rect(
            win, BLACK, (self.x, self.y, self.width, self.height), 2)
        pygame.draw.rect(
            win, self.color, (self.x + 2, self.y + 2, self.width * self.bar_value, self.height - 4))