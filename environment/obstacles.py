
import numpy as np
from dataclasses import dataclass


@dataclass
class Obstacle:
    x: float
    y: float
    radius: float
    obstacle_type: str = "static"

    @property
    def position(self):
        return np.array([self.x, self.y], dtype=np.float32)


class DynamicObstacle(Obstacle):

    def __init__(
        self,
        x,
        y,
        radius,
        speed=0.05,
        obstacle_type="dynamic",
        movement="random",
        bounds=(0, 20, 0, 20)
    ):
        super().__init__(x, y, radius, obstacle_type)

        self.speed = speed
        self.movement = movement
        self.bounds = bounds

        self.velocity = np.zeros(2, dtype=np.float32)
        self.direction = np.random.uniform(0, 2 * np.pi)

        self.change_direction()
        if self.movement == "horizontal":
            self.velocity = np.array(
                [self.speed, 0],
                dtype=np.float32
            )


    def change_direction(self):
        angle = np.random.uniform(0, 2 * np.pi)

        self.velocity = np.array([
            np.cos(angle),
            np.sin(angle)
        ], dtype=np.float32) * self.speed

    def update(self, dt=1.0):

        if self.movement == "random":
            # Occasionally change direction
            if np.random.random() < 0.05:
                self.change_direction()

        elif self.movement == "horizontal":
            self.velocity = np.array(
                [self.speed, 0], dtype=np.float32
            )

        elif self.movement == "vertical":
            self.velocity = np.array(
                [0, self.speed], dtype=np.float32
            )

        # Update position
        self.x += self.velocity[0] * dt
        self.y += self.velocity[1] * dt

        # Boundary collision
        min_x, max_x, min_y, max_y = self.bounds

        if self.x - self.radius < min_x or self.x + self.radius > max_x:
            self.velocity[0] *= -1

        if self.y - self.radius < min_y or self.y + self.radius > max_y:
            self.velocity[1] *= -1

        self.x = np.clip(self.x, min_x + self.radius, max_x - self.radius)
        self.y = np.clip(self.y, min_y + self.radius, max_y - self.radius)


class Human(DynamicObstacle):

    def __init__(self, x, y, bounds):
        super().__init__(
            x=x,
            y=y,
            radius=0.3,
            speed=0.8,
            obstacle_type="human",
            movement="random",
            bounds=bounds
        )


class MovingCrate(DynamicObstacle):

    def __init__(self, x, y, bounds):
        super().__init__(
            x=x,
            y=y,
            radius=0.4,
            speed=0.5,
            obstacle_type="crate",
            movement="horizontal",
            bounds=bounds
        )


class OtherRobot(DynamicObstacle):

    def __init__(self, x, y, bounds):
        super().__init__(
            x=x,
            y=y,
            radius=0.5,
            speed=0.7,
            obstacle_type="robot",
            movement="random",
            bounds=bounds
        )


def check_collision(obj1, obj2):
    distance = np.linalg.norm(
        obj1.position - obj2.position
    )

    return distance < (obj1.radius + obj2.radius)
