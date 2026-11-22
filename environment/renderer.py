
import pygame
import numpy as np


class WarehouseRenderer:

    # Window configuration
    WIDTH = 900
    HEIGHT = 700
    CELL_SIZE = 30

    # Colors
    COLORS = {
        "background": (25, 25, 30),
        "grid": (55, 55, 60),
        "static": (120, 120, 120),
        "human": (40, 200, 100),
        "crate": (230, 180, 40),
        "robot": (160, 80, 220),
        "agent": (50, 140, 255),
        "text": (255, 255, 255),
        "collision": (255, 50, 50)
    }

    def __init__(self, warehouse_size=(20, 20)):

        pygame.init()

        self.width = self.WIDTH
        self.height = self.HEIGHT

        self.screen = pygame.display.set_mode(
            (self.width, self.height)
        )

        pygame.display.set_caption(
            "Autonomous Warehouse Simulation"
        )

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 18)

        self.warehouse_width = warehouse_size[0]
        self.warehouse_height = warehouse_size[1]

        self.scale_x = self.width / self.warehouse_width
        self.scale_y = self.height / self.warehouse_height

        self.running = True

    def world_to_screen(self, position):

        x = int(position[0] * self.scale_x)
        y = int(position[1] * self.scale_y)

        return x, y

    def draw_grid(self):

        for x in range(0, self.width, self.CELL_SIZE):
            pygame.draw.line(
                self.screen,
                self.COLORS["grid"],
                (x, 0),
                (x, self.height)
            )

        for y in range(0, self.height, self.CELL_SIZE):
            pygame.draw.line(
                self.screen,
                self.COLORS["grid"],
                (0, y),
                (self.width, y)
            )

    def draw_obstacle(self, obstacle):

        position = self.world_to_screen(obstacle.position)

        radius = max(
            3,
            int(obstacle.radius * self.scale_x)
        )

        color = self.COLORS.get(
            obstacle.obstacle_type,
            self.COLORS["static"]
        )

        pygame.draw.circle(
            self.screen,
            color,
            position,
            radius
        )

    def draw_agent(self, position, radius=0.5):

        screen_position = self.world_to_screen(position)

        pygame.draw.circle(
            self.screen,
            self.COLORS["agent"],
            screen_position,
            int(radius * self.scale_x)
        )

    def draw_destination(self, position):

        screen_position = self.world_to_screen(position)

        pygame.draw.circle(
            self.screen,
            (255, 255, 255),
            screen_position,
            8,
            2
        )

    def draw_information(self, obstacles, step):

        info = [
            f"Simulation Step: {step}",
            f"Dynamic Objects: {len(obstacles)}"
        ]

        for index, item in enumerate(info):

            text = self.font.render(
                item,
                True,
                self.COLORS["text"]
            )

            self.screen.blit(
                text,
                (10, 10 + index * 25)
            )

    def render(
        self,
        obstacles,
        agent_position=None,
        destination=None,
        step=0
    ):

        if not self.running:
            return

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                self.running = False

        self.screen.fill(self.COLORS["background"])

        self.draw_grid()

        # Draw dynamic and static obstacles
        for obstacle in obstacles:
            self.draw_obstacle(obstacle)

        # Draw autonomous robot
        if agent_position is not None:
            self.draw_agent(agent_position)

        # Draw destination
        if destination is not None:
            self.draw_destination(destination)

        self.draw_information(obstacles, step)

        pygame.display.flip()

        self.clock.tick(60)

    def close(self):

        pygame.quit()
        self.running = False
