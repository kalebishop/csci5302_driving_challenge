import numpy as np
from controller import Display, Robot

LIGHT_GRAY = "0x505050"
RED = "0xBB2222"
GREEN = "0x22BB11"
BLUE = "0x2222BB"

class AVTelemetry:
    def __init__(self, robot):
        self.display = robot.getDisplay("tele_display")
        self.display.setColor(LIGHT_GRAY)

    def create_display(self, data):
        # self.base_image = self.imageNew(data, Display.ARGB, self.display.getWidth, self.display.getHeight)
        pass