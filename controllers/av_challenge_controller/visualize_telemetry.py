import numpy as np
from controller import Display, Robot

WHITE = 0xFFFFFF
LIGHT_GRAY = 0x505050
RED = 0xBB2222
GREEN = 0x22BB11
BLUE = 0x2222BB

class AVTelemetry:
    def __init__(self, robot):
        self.display = robot.getDisplay("tele_display")

        self.width = self.display.getWidth()
        self.height = self.display.getHeight()

        self.X_OFFSET = self.width // 2
        self.Y_OFFSET = self.height // 4

        self.corner_display = (self.width // 10 * 6, self.height // 20 * 19)
        self.display.setFont("Arial", 6, True)

    def reset_display(self):
        self.display.setColor(WHITE)
        self.display.fillRectangle(0, 0, self.width, self.height)

    def display_particles(self, particles):
        # reset
        self.reset_display()
        for p in particles:
            self.display.setColor(BLUE)
            x, y = p.mu[:2]
            self.display.drawRectangle(int(x)+self.X_OFFSET, int(y)+self.Y_OFFSET, 3, 3)
            self.display_landmarks(p.landmarks)

    def display_landmarks(self, landmarks):
        self.display.setColor(GREEN)
        for l in landmarks:
            x, y = (l.mu)
            self.display.drawPixel(int(x)+self.X_OFFSET, int(y)+self.Y_OFFSET)
            # self.display.drawText(str(l.id), int(x), int(y))

    def display_statistics(self, data):
        # data should be a dict of key: value attributions
        self.display.setColor(RED)
        x, y = self.corner_display
        display_string = ""
        for key, val in data.items():
            display_string += (key + " : " + val + "\n")
        if display_string:
            self.display.drawText(display_string, x, y)