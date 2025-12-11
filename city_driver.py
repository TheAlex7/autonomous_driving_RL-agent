# changes made:
#   completely redesigned _create_track()
#   added goal checking in step() for early episode returns
#   changed to discrete action state - dare
#       Added Goal intersection coloring (always yellow, no normalization)
#   car is penalized for driving on grass.
__credits__ = ["Andrea PIERRÉ"]

import math
from typing import List
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.box2d.car_dynamics import Car
from gymnasium.error import DependencyNotInstalled, InvalidAction
from gymnasium.utils import EzPickle


try:
    import Box2D
    from Box2D.b2 import contactListener, fixtureDef, polygonShape
except ImportError as e:
    raise DependencyNotInstalled(
        'Box2D is not installed, you can install it by run `pip install swig` followed by `pip install "gymnasium[box2d]"`'
    ) from e

try:
    # As pygame is necessary for using the environment (reset and step) even without a render mode
    #   therefore, pygame is a necessary import for the environment.
    import pygame
    from pygame import gfxdraw
except ImportError as e:
    raise DependencyNotInstalled(
        'pygame is not installed, run `pip install "gymnasium[box2d]"`'
    ) from e


STATE_W = 96  # less than Atari 160x192
STATE_H = 96
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

SCALE = 6.0  # Track scale
TRACK_RAD = 900 / SCALE  # Track is heavily morphed circle with this radius
PLAYFIELD = 2000 / SCALE  # Game over boundary
FPS = 50  # Frames per second
ZOOM = 2.7  # Camera zoom
ZOOM_FOLLOW = True  # Set to False for fixed view (don't use zoom)


TRACK_DETAIL_STEP = 21 / SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 40 / SCALE
BORDER = 8 / SCALE
BORDER_MIN_COUNT = 4
GRASS_DIM = PLAYFIELD / 20.0
MAX_SHAPE_DIM = (
    max(GRASS_DIM, TRACK_WIDTH, TRACK_DETAIL_STEP) * math.sqrt(2) * ZOOM * SCALE
)


class FrictionDetector(contactListener):
    def __init__(self, env, lap_complete_percent):
        contactListener.__init__(self)
        self.env = env
        self.lap_complete_percent = lap_complete_percent

    def BeginContact(self, contact):
        self._contact(contact, True)

    def EndContact(self, contact):
        self._contact(contact, False)

    def _contact(self, contact, begin):
        tile = None
        obj = None
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        if u1 and "road_friction" in u1.__dict__:
            tile = u1
            obj = u2
        if u2 and "road_friction" in u2.__dict__:
            tile = u2
            obj = u1
        if not tile:
            return

        # inherit tile color from env
        tile.color[:] = self.env.road_color
        if not obj or "tiles" not in obj.__dict__:
            return
        if begin:
            obj.tiles.add(tile)
            if not tile.road_visited:
                tile.road_visited = True
                self.env.reward += 1000.0 / len(self.env.track)
                self.env.tile_visited_count += 1

                # Lap is considered completed if enough % of the track was covered
                if (
                    tile.idx == 0
                    and self.env.tile_visited_count / len(self.env.track)
                    > self.lap_complete_percent
                ):
                    self.env.new_lap = True
        else:
            obj.tiles.remove(tile)


class CityDrive(gym.Env, EzPickle):
    """
    ## Description
    The easiest control task to learn from pixels - a top-down
    racing environment. The generated track is random every episode.

    Some indicators are shown at the bottom of the window along with the
    state RGB buffer. From left to right: true speed, four ABS sensors,
    steering wheel position, and gyroscope.
    To play yourself (it's rather fast for humans), type:
    ```shell
    python gymnasium/envs/box2d/car_racing.py
    ```
    Remember: it's a powerful rear-wheel drive car - don't press the accelerator
    and turn at the same time.

    ## Action Space
    If continuous there are 3 actions :
    - 0: steering, -1 is full left, +1 is full right
    - 1: gas
    - 2: braking

    If discrete there are 5 actions:
    - 0: do nothing
    - 1: steer right
    - 2: steer left
    - 3: gas
    - 4: brake

    ## Observation Space

    A top-down 96x96 RGB image of the car and race track.

    ## Rewards
    The reward is -0.1 every frame and +1000/N for every track tile visited, where N is the total number of tiles
     visited in the track. For example, if you have finished in 732 frames, your reward is 1000 - 0.1*732 = 926.8 points.

    ## Starting State
    The car starts at rest in the center of the road.

    ## Episode Termination
    The episode finishes when all the tiles are visited. The car can also go outside the playfield -
     that is, far off the track, in which case it will receive -100 reward and die.

    ## Arguments

    ```python
    >>> import gymnasium as gym
    >>> env = gym.make("CityDrive-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False, continuous=False)
    >>> env
    <TimeLimit<OrderEnforcing<PassiveEnvChecker<CityDrive<CityDrive-v3>>>>>

    ```

    * `lap_complete_percent=0.95` dictates the percentage of tiles that must be visited by
     the agent before a lap is considered complete.

    * `domain_randomize=False` enables the domain randomized variant of the environment.
     In this scenario, the background and track colours are different on every reset.

    * `continuous=True` specifies if the agent has continuous (true) or discrete (false) actions.
     See action space section for a description of each.

    ## Reset Arguments

    Passing the option `options["randomize"] = True` will change the current colour of the environment on demand.
    Correspondingly, passing the option `options["randomize"] = False` will not change the current colour of the environment.
    `domain_randomize` must be `True` on init for this argument to work.

    ```python
    >>> import gymnasium as gym
    >>> env = gym.make("CityDrive-v3", domain_randomize=True)

    # normal reset, this changes the colour scheme by default
    >>> obs, _ = env.reset()

    # reset with colour scheme change
    >>> randomize_obs, _ = env.reset(options={"randomize": True})

    # reset with no colour scheme change
    >>> non_random_obs, _ = env.reset(options={"randomize": False})

    ```

    ## Version History
    - v2: Change truncation to termination when finishing the lap (1.0.0)
    - v1: Change track completion logic and add domain randomization (0.24.0)
    - v0: Original version

    ## References
    - Chris Campbell (2014), http://www.iforce2d.net/b2dtut/top-down-car.

    ## Credits
    Created by Oleg Klimov
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "state_pixels",
        ],
        "render_fps": FPS,
    }

    def __init__(
        self,
        render_mode: str | None = None,
        verbose: bool = False,
        lap_complete_percent: float = 1.0,
        domain_randomize: bool = False,
        continuous: bool = False,  # darius edit to be DQN
        num_streets: int = 4,
    ):
        EzPickle.__init__(
            self,
            render_mode,
            verbose,
            lap_complete_percent,
            domain_randomize,
            continuous,
            num_streets,
        )
        self.continuous = continuous
        self.domain_randomize = domain_randomize
        self.lap_complete_percent = lap_complete_percent
        self._init_colors()

        # Added With Claude
        self.max_episode_steps = 1000
        self.steps_taken = 0
        self.contactListener_keepref = FrictionDetector(self, self.lap_complete_percent)
        self.world = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)
        self.screen: pygame.Surface | None = None
        self.surf = None
        self.clock = None
        self.isopen = True
        self.invisible_state_window = None
        self.invisible_video_window = None
        self.road = None
        self.car: Car | None = None
        self.reward = 0.0
        self.prev_reward = 0.0
        self.verbose = verbose
        self.new_lap = False
        self.fd_tile = fixtureDef(
            shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)])
        )
        self.end_pos: tuple[int, int] | None = None
        # list of coordinates for goal intersection polygon. Default none
        self.end_poly: (
            List[
                tuple[np.float32, np.float32],
                tuple[np.float32, np.float32],
                tuple[np.float32, np.float32],
                tuple[np.float32, np.float32],
            ]
            | None
        ) = None
        self.old_dist = 0
        if num_streets < 2:
            self.vertical_streets = 2
            self.horizontal_streets = 2
        else:
            self.vertical_streets = num_streets
            self.horizontal_streets = num_streets

        # This will throw a warning in tests/envs/test_envs in utils/env_checker.py as the space is not symmetric
        #   or normalised however this is not possible here so ignore
        if self.continuous:
            self.action_space = spaces.Box(
                np.array([-1, 0, 0]).astype(np.float32),
                np.array([+1, +1, +1]).astype(np.float32),
            )  # steer, gas, brake
        else:
            self.action_space = spaces.Discrete(5)
            # do nothing, right, left, gas, brake

        self.observation_space = spaces.Dict(
            {
                "image_array": spaces.Box(
                    low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8
                ),  # 255 x 255 image
                "agent_loc": spaces.Box(
                    low=-PLAYFIELD, high=PLAYFIELD, shape=(2,), dtype=np.float32
                ),  # [x, y] coordinates
                "target_loc": spaces.Box(
                    low=-PLAYFIELD, high=PLAYFIELD, shape=(2,), dtype=np.float32
                ),  # [x, y] coordinates
            }
        )

        self.render_mode = render_mode

    def _destroy(self):
        if not self.road:
            return
        for t in self.road:
            self.world.DestroyBody(t)
        self.road = []
        assert self.car is not None
        self.car.destroy()

    def _init_colors(self):
        if self.domain_randomize:
            # domain randomize the bg and grass colour
            self.road_color = self.np_random.uniform(0, 210, size=3)

            self.bg_color = self.np_random.uniform(0, 210, size=3)

            self.grass_color = np.copy(self.bg_color)
            idx = self.np_random.integers(3)
            self.grass_color[idx] += 20
        else:
            # default colours
            self.road_color = np.array([102, 102, 102])
            self.bg_color = np.array([102, 204, 102])
            self.grass_color = np.array([102, 230, 102])

    def _reinit_colors(self, randomize):
        assert (
            self.domain_randomize
        ), "domain_randomize must be True to use this function."

        if randomize:
            # domain randomize the bg and grass colour
            self.road_color = self.np_random.uniform(0, 210, size=3)

            self.bg_color = self.np_random.uniform(0, 210, size=3)

            self.grass_color = np.copy(self.bg_color)
            idx = self.np_random.integers(3)
            self.grass_color[idx] += 20

    # check if a coordinate is inside a polygon
    def _point_in_poly(self, x, y, poly):
        # each polygon is made of 4 corners, each corner location is given as xy-coordinates
        inside = False
        n = len(poly)  # all road polygons will have 4 points only, so n=4 typically
        p1x, p1y = poly[0]
        for i in range(n + 1):
            p2x, p2y = poly[i % n]
            if ((p1y > y) != (p2y > y)) and (
                x < (p2x - p1x) * (y - p1y) / (p2y - p1y) + p1x
            ):
                inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def _create_track(self):

        # Generate a square city-grid.
        # Start is (0,0) and Goal is random intersection.

        GRID_W = self.vertical_streets  # number of vertical streets
        GRID_H = self.horizontal_streets  # horizontal streets
        BLOCK = 40  # grid spacing (size of each block)
        ROAD_WIDTH = TRACK_WIDTH

        intersections = []
        for iy in range(GRID_H):
            for ix in range(GRID_W):
                x = ix * BLOCK
                y = iy * BLOCK
                intersections.append((ix, iy, x, y))

        # Starting point is always (0,0)
        _, _, start_x, start_y = intersections[0]

        # Random intersection not including start (0,0)
        rand_idx = self.np_random.integers(1, len(intersections))
        _, _, goal_x, goal_y = intersections[rand_idx]

        # road segments for entire grid
        self.road = []
        self.road_poly = []  # vertices for road polygons + color
        road_bodies = []

        self.track = []  # race track

        # horizontal streets
        for iy in range(GRID_H):
            for ix in range(GRID_W - 1):
                x1 = ix * BLOCK
                y1 = iy * BLOCK
                x2 = (ix + 1) * BLOCK
                y2 = y1

                OVERLAP = ROAD_WIDTH * 0.5
                x1 -= OVERLAP
                x2 += OVERLAP

                # heading direction
                beta = math.atan2(y2 - y1, x2 - x1)

                # left/right offsets
                left1 = (
                    x1 - ROAD_WIDTH * math.sin(beta),
                    y1 + ROAD_WIDTH * math.cos(beta),
                )
                right1 = (
                    x1 + ROAD_WIDTH * math.sin(beta),
                    y1 - ROAD_WIDTH * math.cos(beta),
                )
                left2 = (
                    x2 - ROAD_WIDTH * math.sin(beta),
                    y2 + ROAD_WIDTH * math.cos(beta),
                )
                right2 = (
                    x2 + ROAD_WIDTH * math.sin(beta),
                    y2 - ROAD_WIDTH * math.cos(beta),
                )

                vertices = [left1, right1, right2, left2]

                # Create static road tile
                self.fd_tile.shape.vertices = vertices
                tile = self.world.CreateStaticBody(fixtures=self.fd_tile)
                tile.userData = tile
                tile.color = self.road_color
                tile.road_visited = False
                tile.road_friction = 1.0
                tile.fixtures[0].sensor = True
                tile.idx = len(road_bodies)

                road_bodies.append(tile)
                self.road.append(tile)
                self.road_poly.append((vertices, tile.color))

        # vertical streets
        for ix in range(GRID_W):
            for iy in range(GRID_H - 1):
                x1 = ix * BLOCK
                y1 = iy * BLOCK
                x2 = x1
                y2 = (iy + 1) * BLOCK

                OVERLAP = ROAD_WIDTH * 0.5
                y1 -= OVERLAP
                y2 += OVERLAP

                beta = math.atan2(y2 - y1, x2 - x1)

                left1 = (
                    x1 - ROAD_WIDTH * math.sin(beta),
                    y1 + ROAD_WIDTH * math.cos(beta),
                )
                right1 = (
                    x1 + ROAD_WIDTH * math.sin(beta),
                    y1 - ROAD_WIDTH * math.cos(beta),
                )
                left2 = (
                    x2 - ROAD_WIDTH * math.sin(beta),
                    y2 + ROAD_WIDTH * math.cos(beta),
                )
                right2 = (
                    x2 + ROAD_WIDTH * math.sin(beta),
                    y2 - ROAD_WIDTH * math.cos(beta),
                )

                vertices = [left1, right1, right2, left2]

                self.fd_tile.shape.vertices = vertices
                tile = self.world.CreateStaticBody(fixtures=self.fd_tile)
                tile.userData = tile
                tile.color = self.road_color
                tile.road_visited = False
                tile.road_friction = 1.0
                tile.fixtures[0].sensor = True
                tile.idx = len(road_bodies)

                road_bodies.append(tile)
                self.road.append(tile)
                self.road_poly.append((vertices, tile.color))

        # Build track entries for intersections (one entry per intersection)
        for ix, iy, x, y in intersections:
            # choose an arbitrary orientation based on neighbors
            if ix < GRID_W - 1:
                nx, ny = (ix + 1) * BLOCK, iy * BLOCK
                beta = math.atan2(ny - y, nx - x)
            else:
                nx, ny = x, (iy + 1) * BLOCK if iy < GRID_H - 1 else y
                beta = math.atan2(ny - y, nx - x)

            alpha = 0.0  # unused
            self.track.append((alpha, beta, x, y))

        self.start_pos = (start_x, start_y)
        self.end_pos = (goal_x, goal_y)
        self.old_dist = np.sqrt(
            (start_x - goal_x) ** 2 + (start_y - goal_y) ** 2
        )  # inital distance from start to end

        # Define Goal Intersection Polygon
        GOAL_SIZE = TRACK_WIDTH  # half-size of the square goal area

        left1 = (goal_x - GOAL_SIZE, goal_y + GOAL_SIZE)
        right1 = (goal_x + GOAL_SIZE, goal_y + GOAL_SIZE)
        right2 = (goal_x + GOAL_SIZE, goal_y - GOAL_SIZE)
        left2 = (goal_x - GOAL_SIZE, goal_y - GOAL_SIZE)

        self.end_poly = [left1, right1, right2, left2]

        return True

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        super().reset(seed=seed)
        self._destroy()
        self.world.contactListener_bug_workaround = FrictionDetector(
            self, self.lap_complete_percent
        )
        self.steps_taken = 0
        self.world.contactListener = self.world.contactListener_bug_workaround
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.t = 0.0
        self.new_lap = False
        self.road_poly = []
        self.end_pos: tuple[int, int] | None = None
        # list of coordinates for goal intersection polygon. Default none
        self.end_poly: (
            List[
                tuple[np.float32, np.float32],
                tuple[np.float32, np.float32],
                tuple[np.float32, np.float32],
                tuple[np.float32, np.float32],
            ]
            | None
        ) = None

        if self.domain_randomize:
            randomize = True
            if isinstance(options, dict):
                if "randomize" in options:
                    randomize = options["randomize"]

            self._reinit_colors(randomize)

        while True:
            success = self._create_track()
            if success:
                break
            if self.verbose:
                print(
                    "retry to generate track (normal if there are not many"
                    "instances of this message)"
                )
        self.car = Car(self.world, *self.track[0][1:4])

        if self.render_mode == "human":
            self.render()
        return self.step(None)[0], {}

    def step(self, action: np.ndarray | int):
        assert self.car is not None
        if action is not None:
            if self.continuous:
                action = action.astype(np.float64)
                self.car.steer(-action[0])
                self.car.gas(action[1])
                self.car.brake(action[2])
            else:
                if not self.action_space.contains(action):
                    raise InvalidAction(
                        f"you passed the invalid action `{action}`. "
                        f"The supported action_space is `{self.action_space}`"
                    )
                self.car.steer(-1.1 * (action == 1) + 1.1 * (action == 2))
                self.car.gas(0.2 * (action == 3))
                self.car.brake(0.8 * (action == 4))

        self.car.step(1.0 / FPS)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS

        self.state = self._render("state_pixels")

        self.steps_taken += 1
        # added via Claude to create a time urgency
        step_reward = 0
        terminated = False
        truncated = False
        info = {}
        car_x, car_y = self.car.hull.position
        # if (action is not None and action is not 0 and action is not 4):
        # First step without action, called from reset() also makes sure its moving
        # make sure its not just continuously braking

        # Penalty if off road
        on_road = any(
            self._point_in_poly(car_x, car_y, poly) for poly, _ in self.road_poly
        )

        speed = np.sqrt(
            self.car.hull.linearVelocity[0] ** 2 + self.car.hull.linearVelocity[1] ** 2
        )

        # Small push for spending too much time
        step_reward -= 0.05  # Claude: bullying :3

        # Distance to goal (shaping)
        if self.end_pos is not None:
            new_dist = np.sqrt(
                (car_x - self.end_pos[0]) ** 2 + (car_y - self.end_pos[1]) ** 2
            )
            # step_reward += (self.old_dist - new_dist) * 0.5  # reward for progress
            progress = self.old_dist - new_dist
            # Only reward progress if ON ROAD
            if on_road:
                step_reward += progress * 1.0  # Strong reward for progress
            else:
                # Heavily penalize off-road progress (no shortcuts!)
                step_reward += progress * 0.1  # 10x less reward off-road

            self.old_dist = new_dist

        # Velocity toward goal (encourages moving in right direction)
        velocity_toward_goal = self._get_velocity_toward_goal()
        # step_reward += velocity_toward_goal * 0.05

        if on_road:
            step_reward += velocity_toward_goal * 0.1
        # step_reward -= 1.0  # changed from -10 then 0.5
        # else:
        #     pass
        # step_reward += 0.05
        # # Only reward being on road if MOVING toward goal
        # if velocity_toward_goal > 0.5:  # Only if moving forward
        #     step_reward += 0.1

        # Road penalties/rewards - MAKE OFF-ROAD VERY EXPENSIVE
        if not on_road:
            # Massive penalty that scales with speed (discourage shortcuts)
            step_reward -= 2.0  # Base penalty
            step_reward -= speed * 0.5  # Extra penalty for fast off-roading
        else:
            # Reward for staying on road
            step_reward += 0.1

        # Penalize excessive steering or braking (smooth driving)
        if action == 1 or action == 2:  # steering
            step_reward -= 0.01
        # Reward for using gas (encourage forward movement)
        if action == 3:  # gas
            step_reward += 0.05
        if action == 4:  # braking
            step_reward -= 0.1  # changed from 0.02
            if speed < 0.5:  # Almost stopped
                step_reward -= 0.3  # Severe penalty for sitting still with brake

        # Add timeout truncation
        if self.steps_taken >= self.max_episode_steps:
            truncated = True
            step_reward -= 20  # Small penalty for timeout, changed from 10

        self.prev_reward = step_reward

        # Goal reached
        # goal_radius = TRACK_WIDTH * 4  # check
        # if new_dist < goal_radius:
        #     step_reward += 100
        #     # changed from +200 to +50
        #     terminated = True

        # Goal reached - check if inside the yellow goal polygon
        if self.end_pos is not None and self.end_poly is not None:
            if self._point_in_poly(car_x, car_y, self.end_poly):
                step_reward += 100  # Increased from 50 for stronger signal
                terminated = True
                info["goal_reached"] = True

        # if self.tile_visited_count == len(self.track) or self.new_lap:
        #     # Termination due to finishing lap
        #     terminated = True
        #     info["lap_finished"] = True
        x, y = self.car.hull.position
        if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
            terminated = True
            info["lap_finished"] = False
            step_reward = -100
            # changed from -100 to -50
        # else:
        #     step_reward -= 1.5  # MOOOOOOVE DO SOMETHING

        if self.render_mode == "human":
            self.render()

        # print(self.car.hull.position[0],self.car.hull.position[1])
        return self.state, step_reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        else:
            return self._render(self.render_mode)

    def _get_velocity_toward_goal(self):
        """Calculate component of velocity toward goal"""
        if self.end_pos is None:
            return 0

        car_x, car_y = self.car.hull.position
        vel_x, vel_y = self.car.hull.linearVelocity

        # Direction to goal
        goal_dir_x = self.end_pos[0] - car_x
        goal_dir_y = self.end_pos[1] - car_y
        dist = np.sqrt(goal_dir_x**2 + goal_dir_y**2)

        if dist > 0:
            goal_dir_x /= dist
            goal_dir_y /= dist

        # Dot product: velocity projected onto goal direction
        return vel_x * goal_dir_x + vel_y * goal_dir_y

    def _render(self, mode: str):
        assert mode in self.metadata["render_modes"]

        pygame.font.init()
        if self.screen is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        if "t" not in self.__dict__:
            return  # reset() not called yet

        self.surf = pygame.Surface((WINDOW_W, WINDOW_H))

        assert self.car is not None
        # computing transformations
        angle = -self.car.hull.angle
        # Animating first second zoom.
        zoom = 0.1 * SCALE * max(1 - self.t, 0) + ZOOM * SCALE * min(self.t, 1)
        scroll_x = -(self.car.hull.position[0]) * zoom
        scroll_y = -(self.car.hull.position[1]) * zoom
        trans = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(angle)
        trans = (WINDOW_W / 2 + trans[0], WINDOW_H / 4 + trans[1])

        self._render_road(zoom, trans, angle)
        self.car.draw(
            self.surf,
            zoom,
            trans,
            angle,
            mode not in ["state_pixels_list", "state_pixels"],
        )

        self.surf = pygame.transform.flip(self.surf, False, True)

        # showing stats
        self._render_indicators(WINDOW_W, WINDOW_H)

        font = pygame.font.Font(pygame.font.get_default_font(), 42)
        text = font.render("%04i" % self.reward, True, (255, 255, 255), (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (60, WINDOW_H - WINDOW_H * 2.5 / 40.0)
        self.surf.blit(text, text_rect)

        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            assert self.screen is not None
            self.screen.fill(0)
            self.screen.blit(self.surf, (0, 0))
            pygame.display.flip()
        elif mode == "rgb_array":
            return {
                "image_array": self._create_image_array(self.surf, (VIDEO_W, VIDEO_H)),
                "agent_loc": (self.car.hull.position[0], self.car.hull.position[1]),
                "target_loc": self.end_pos,
            }
        elif mode == "state_pixels":
            return {
                "image_array": self._create_image_array(self.surf, (STATE_W, STATE_H)),
                "agent_loc": (self.car.hull.position[0], self.car.hull.position[1]),
                "target_loc": self.end_pos,
            }
        else:
            return self.isopen

    def _render_road(self, zoom, translation, angle):
        bounds = PLAYFIELD
        field = [
            (bounds, bounds),
            (bounds, -bounds),
            (-bounds, -bounds),
            (-bounds, bounds),
        ]

        # draw background
        self._draw_colored_polygon(
            self.surf, field, self.bg_color, zoom, translation, angle, clip=False
        )

        # draw grass patches
        grass = []
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                grass.append(
                    [
                        (GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y + 0),
                        (GRASS_DIM * x + 0, GRASS_DIM * y + 0),
                        (GRASS_DIM * x + 0, GRASS_DIM * y + GRASS_DIM),
                        (GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y + GRASS_DIM),
                    ]
                )
        for poly in grass:
            self._draw_colored_polygon(
                self.surf, poly, self.grass_color, zoom, translation, angle
            )

        # draw road
        for poly, color in self.road_poly:
            # converting to pixel coordinates
            poly = [(p[0], p[1]) for p in poly]
            color = [int(c) for c in color]
            self._draw_colored_polygon(self.surf, poly, color, zoom, translation, angle)

        # coloring goal intersection yellow
        c_yellow = np.array([255, 255, 0])
        self._draw_colored_polygon(
            self.surf, self.end_poly, c_yellow, zoom, translation, angle
        )

    def _render_indicators(self, W, H):
        s = W / 40.0
        h = H / 40.0
        color = (0, 0, 0)
        polygon = [(W, H), (W, H - 5 * h), (0, H - 5 * h), (0, H)]
        pygame.draw.polygon(self.surf, color=color, points=polygon)

        def vertical_ind(place, val):
            return [
                (place * s, H - (h + h * val)),
                ((place + 1) * s, H - (h + h * val)),
                ((place + 1) * s, H - h),
                ((place + 0) * s, H - h),
            ]

        def horiz_ind(place, val):
            return [
                ((place + 0) * s, H - 4 * h),
                ((place + val) * s, H - 4 * h),
                ((place + val) * s, H - 2 * h),
                ((place + 0) * s, H - 2 * h),
            ]

        assert self.car is not None
        true_speed = np.sqrt(
            np.square(self.car.hull.linearVelocity[0])
            + np.square(self.car.hull.linearVelocity[1])
        )

        # simple wrapper to render if the indicator value is above a threshold
        def render_if_min(value, points, color):
            if abs(value) > 1e-4:
                pygame.draw.polygon(self.surf, points=points, color=color)

        render_if_min(true_speed, vertical_ind(5, 0.02 * true_speed), (255, 255, 255))
        # ABS sensors
        render_if_min(
            self.car.wheels[0].omega,
            vertical_ind(7, 0.01 * self.car.wheels[0].omega),
            (0, 0, 255),
        )
        render_if_min(
            self.car.wheels[1].omega,
            vertical_ind(8, 0.01 * self.car.wheels[1].omega),
            (0, 0, 255),
        )
        render_if_min(
            self.car.wheels[2].omega,
            vertical_ind(9, 0.01 * self.car.wheels[2].omega),
            (51, 0, 255),
        )
        render_if_min(
            self.car.wheels[3].omega,
            vertical_ind(10, 0.01 * self.car.wheels[3].omega),
            (51, 0, 255),
        )

        render_if_min(
            self.car.wheels[0].joint.angle,
            horiz_ind(20, -10.0 * self.car.wheels[0].joint.angle),
            (0, 255, 0),
        )
        render_if_min(
            self.car.hull.angularVelocity,
            horiz_ind(30, -0.8 * self.car.hull.angularVelocity),
            (255, 0, 0),
        )

    def _draw_colored_polygon(
        self, surface, poly, color, zoom, translation, angle, clip=True
    ):
        poly = [pygame.math.Vector2(c).rotate_rad(angle) for c in poly]
        poly = [
            (c[0] * zoom + translation[0], c[1] * zoom + translation[1]) for c in poly
        ]
        # This checks if the polygon is out of bounds of the screen, and we skip drawing if so.
        # Instead of calculating exactly if the polygon and screen overlap,
        # we simply check if the polygon is in a larger bounding box whose dimension
        # is greater than the screen by MAX_SHAPE_DIM, which is the maximum
        # diagonal length of an environment object
        if not clip or any(
            (-MAX_SHAPE_DIM <= coord[0] <= WINDOW_W + MAX_SHAPE_DIM)
            and (-MAX_SHAPE_DIM <= coord[1] <= WINDOW_H + MAX_SHAPE_DIM)
            for coord in poly
        ):
            gfxdraw.aapolygon(self.surf, poly, color)
            gfxdraw.filled_polygon(self.surf, poly, color)

    def _create_image_array(self, screen, size):
        scaled_screen = pygame.transform.smoothscale(screen, size)
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(1, 0, 2)
        )

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            self.isopen = False
            pygame.quit()


if __name__ == "__main__":
    a = np.array([0.0, 0.0, 0.0])

    def register_input():
        global quit, restart
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    a[0] = -1.0
                if event.key == pygame.K_RIGHT:
                    a[0] = +1.0
                if event.key == pygame.K_UP:
                    a[1] = +1.0
                if event.key == pygame.K_DOWN:
                    a[2] = +0.8  # set 1.0 for wheels to block to zero rotation
                if event.key == pygame.K_RETURN:
                    restart = True
                if event.key == pygame.K_ESCAPE:
                    quit = True

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    a[0] = 0
                if event.key == pygame.K_RIGHT:
                    a[0] = 0
                if event.key == pygame.K_UP:
                    a[1] = 0
                if event.key == pygame.K_DOWN:
                    a[2] = 0

            if event.type == pygame.QUIT:
                quit = True

    env = CityDrive(render_mode="human", num_streets=4)

    quit = False
    while not quit:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            register_input()
            s, r, terminated, truncated, info = env.step(a)
            total_reward += r
            if steps % 200 == 0 or terminated or truncated:
                print("\naction " + str([f"{x:+0.2f}" for x in a]))
                print(f"step {steps} total_reward {total_reward:+0.2f}")
            steps += 1
            if terminated or truncated or restart or quit:
                break
    env.close()
