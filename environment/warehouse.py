
import gymnasium as gym
import numpy as np

from gymnasium import spaces

from environment.obstacles import (
    Obstacle,
    Human,
    MovingCrate,
    OtherRobot,
)


class WarehouseEnv(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        render_mode=None,
        warehouse_size=(20, 20),
        max_steps=1000
    ):
        super().__init__()

        self.width, self.height = warehouse_size
        self.render_mode = render_mode
        self.max_steps = max_steps

        # Robot configuration
        self.robot_radius = 0.35
        self.max_linear_velocity = 1.2
        self.max_angular_velocity = 2.5
        self.dt = 0.1

        # Sensor configuration
        self.lidar_rays = 16
        self.lidar_range = 5.0
        self.max_tracked_objects = 5

        # Continuous actions:
        # [linear_velocity, angular_velocity]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        # Observation:
        # goal distance + goal angle (sin, cos)
        # robot velocity
        # LiDAR readings
        # 5 nearest objects: relative x, y, vx, vy
        obs_size = 4 + self.lidar_rays + (
            self.max_tracked_objects * 4
        )

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_size,),
            dtype=np.float32
        )

        self.static_obstacles = []
        self.dynamic_obstacles = []

        self.robot_position = np.zeros(2, dtype=np.float32)
        self.robot_heading = 0.0
        self.robot_velocity = 0.0
        self.angular_velocity = 0.0

        self.goal = np.zeros(2, dtype=np.float32)

        self.previous_goal_distance = 0.0
        self.step_count = 0

        self.renderer = None

    # --------------------------------------------------
    # Environment generation
    # --------------------------------------------------

    def _random_position(self, radius=0.5):
        return np.array([
            self.np_random.uniform(radius + 1, self.width - radius - 1),
            self.np_random.uniform(radius + 1, self.height - radius - 1)
        ], dtype=np.float32)

    def _is_position_free(
        self,
        position,
        radius,
        existing,
        min_distance=0.3
    ):
        for obj in existing:
            distance = np.linalg.norm(
                position - obj.position
            )

            if distance < radius + obj.radius + min_distance:
                return False

        return True

    def _spawn_position(self, radius, existing):
        for _ in range(500):
            position = self._random_position(radius)

            if self._is_position_free(
                position, radius, existing
            ):
                return position

        raise RuntimeError("Unable to find free spawn position")

    def _create_warehouse(self):
        self.static_obstacles = []
        self.dynamic_obstacles = []

        # Generate shelf locations
        for _ in range(12):
            position = self._spawn_position(
                0.7, self.static_obstacles
            )

            self.static_obstacles.append(
                Obstacle(
                    x=float(position[0]),
                    y=float(position[1]),
                    radius=0.7,
                    obstacle_type="static"
                )
            )

        # Spawn robot
        self.robot_position = self._spawn_position(
            self.robot_radius,
            self.static_obstacles
        )

        self.robot_heading = self.np_random.uniform(
            -np.pi, np.pi
        )

        # Spawn destination
        for _ in range(500):
            goal = self._random_position(0.4)

            if (
                np.linalg.norm(goal - self.robot_position) > 8
                and self._is_position_free(
                    goal, 0.4, self.static_obstacles
                )
            ):
                self.goal = goal
                break
        else:
            raise RuntimeError("Unable to place destination")

        # Dynamic objects
        bounds = (0, self.width, 0, self.height)

        object_classes = [
            (Human, 3),
            (MovingCrate, 2),
            (OtherRobot, 2)
        ]

        existing = list(self.static_obstacles)

        for object_class, count in object_classes:
            for _ in range(count):
                position = self._spawn_position(
                    0.5, existing
                )

                obj = object_class(
                    float(position[0]),
                    float(position[1]),
                    bounds
                )

                self.dynamic_obstacles.append(obj)
                existing.append(obj)

    # --------------------------------------------------
    # Collision detection
    # --------------------------------------------------

    def _check_robot_collision(self, position):

        # Warehouse boundaries
        if (
            position[0] < self.robot_radius
            or position[0] > self.width - self.robot_radius
            or position[1] < self.robot_radius
            or position[1] > self.height - self.robot_radius
        ):
            return True

        # Static obstacles
        for obstacle in self.static_obstacles:
            distance = np.linalg.norm(
                position - obstacle.position
            )

            if distance < self.robot_radius + obstacle.radius:
                return True

        # Dynamic obstacles
        for obstacle in self.dynamic_obstacles:
            distance = np.linalg.norm(
                position - obstacle.position
            )

            if distance < self.robot_radius + obstacle.radius:
                return True

        return False

    # --------------------------------------------------
    # LiDAR simulation
    # --------------------------------------------------

    def _lidar_scan(self):

        readings = np.ones(
            self.lidar_rays, dtype=np.float32
        )

        obstacles = (
            self.static_obstacles +
            self.dynamic_obstacles
        )

        for i in range(self.lidar_rays):

            angle = (
                self.robot_heading +
                2 * np.pi * i / self.lidar_rays
            )

            direction = np.array([
                np.cos(angle),
                np.sin(angle)
            ])

            nearest_distance = self.lidar_range

            # Check warehouse boundaries
            for axis in range(2):
                if abs(direction[axis]) > 1e-8:
                    boundary = (
                        self.width if axis == 0 else self.height
                    ) if direction[axis] > 0 else 0

                    distance = (
                        boundary - self.robot_position[axis]
                    ) / direction[axis]

                    if distance >= 0:
                        nearest_distance = min(
                            nearest_distance, distance
                        )

            # Ray-circle intersection
            for obstacle in obstacles:

                relative = (
                    obstacle.position - self.robot_position
                )

                projection = np.dot(relative, direction)

                perpendicular_sq = (
                    np.dot(relative, relative) -
                    projection ** 2
                )

                radius_sq = obstacle.radius ** 2

                if perpendicular_sq > radius_sq:
                    continue

                offset = np.sqrt(
                    radius_sq - perpendicular_sq
                )

                distance = projection - offset

                if distance >= 0:
                    nearest_distance = min(
                        nearest_distance, distance
                    )

            readings[i] = np.clip(
                nearest_distance / self.lidar_range,
                0, 1
            )

        return readings

    # --------------------------------------------------
    # Observation builder
    # --------------------------------------------------

    def _get_observation(self):

        goal_vector = self.goal - self.robot_position

        goal_distance = np.linalg.norm(goal_vector)

        goal_angle = np.arctan2(
            goal_vector[1], goal_vector[0]
        ) - self.robot_heading

        goal_angle = np.arctan2(
            np.sin(goal_angle), np.cos(goal_angle)
        )

        observation = [
            goal_distance / np.sqrt(
                self.width ** 2 + self.height ** 2
            ),
            np.sin(goal_angle),
            np.cos(goal_angle),
            self.robot_velocity / self.max_linear_velocity
        ]

        # LiDAR
        observation.extend(self._lidar_scan())

        # Nearest dynamic objects
        objects = sorted(
            self.dynamic_obstacles,
            key=lambda obj: np.linalg.norm(
                obj.position - self.robot_position
            )
        )

        for obj in objects[:self.max_tracked_objects]:

            relative_position = (
                obj.position - self.robot_position
            )

            relative_velocity = obj.velocity.copy()

            observation.extend([
                relative_position[0] / self.width,
                relative_position[1] / self.height,
                relative_velocity[0] / 2.0,
                relative_velocity[1] / 2.0
            ])

        # Padding when fewer objects exist
        while len(observation) < self.observation_space.shape[0]:
            observation.extend([0.0] * 4)

        return np.clip(
            np.asarray(observation, dtype=np.float32),
            -1.0,
            1.0
        )

    # --------------------------------------------------
    # Gymnasium reset
    # --------------------------------------------------

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self._create_warehouse()

        self.robot_velocity = 0.0
        self.angular_velocity = 0.0

        self.step_count = 0

        self.previous_goal_distance = np.linalg.norm(
            self.goal - self.robot_position
        )

        observation = self._get_observation()

        info = {
            "goal_distance": self.previous_goal_distance
        }

        return observation, info

    # --------------------------------------------------
    # Gymnasium step
    # --------------------------------------------------

    def step(self, action):

        self.step_count += 1

        action = np.clip(action, -1.0, 1.0)

        linear_velocity = (
            action[0] * self.max_linear_velocity
        )

        angular_velocity = (
            action[1] * self.max_angular_velocity
        )

        # Update dynamic obstacles
        for obstacle in self.dynamic_obstacles:
            obstacle.update(dt=self.dt)

        # Update robot orientation
        self.robot_heading += angular_velocity * self.dt

        self.robot_heading = np.arctan2(
            np.sin(self.robot_heading),
            np.cos(self.robot_heading)
        )

        # Calculate new robot position
        movement = np.array([
            np.cos(self.robot_heading),
            np.sin(self.robot_heading)
        ]) * linear_velocity * self.dt

        proposed_position = self.robot_position + movement

        collision = self._check_robot_collision(
            proposed_position
        )

        if collision:
            self.robot_velocity = 0.0
        else:
            self.robot_position = proposed_position
            self.robot_velocity = linear_velocity

        self.angular_velocity = angular_velocity

        # Goal distance
        goal_distance = np.linalg.norm(
            self.goal - self.robot_position
        )

        reached_goal = goal_distance < 0.6

        # Reward
        progress = (
            self.previous_goal_distance - goal_distance
        )

        reward = 2.0 * progress
        reward -= 0.05

        if collision:
            reward -= 100.0

        if reached_goal:
            reward += 100.0

        self.previous_goal_distance = goal_distance

        terminated = collision or reached_goal

        truncated = self.step_count >= self.max_steps

        observation = self._get_observation()

        info = {
            "collision": collision,
            "reached_goal": reached_goal,
            "goal_distance": goal_distance,
            "step": self.step_count
        }

        if self.render_mode == "human":
            self.render()

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info
        )

    # --------------------------------------------------
    # Rendering
    # --------------------------------------------------

    def render(self):

        if self.renderer is None:
            from environment.renderer import WarehouseRenderer

            self.renderer = WarehouseRenderer(
                (self.width, self.height)
            )

        all_obstacles = (
            self.static_obstacles +
            self.dynamic_obstacles
        )

        self.renderer.render(
            obstacles=all_obstacles,
            agent_position=self.robot_position,
            destination=self.goal,
            step=self.step_count
        )

    def close(self):

        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
