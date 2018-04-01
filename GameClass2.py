import sys, random
import math
import numpy as np

import pygame
from pygame.locals import *
from pygame.color import THECOLORS

import pymunk
from pymunk.vec2d import Vec2d
import pymunk.pygame_util


# PyGame init
width = 1200
height = 900
pygame.init()
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

# Global variables
sensor_range = 100
robot_radius = 30
robot_velocity = 200
obs_radius = 30


class GameClass:
    def __init__(self):
        # Physics conditions.
        self.space = pymunk.Space()
        self.space.gravity = pymunk.Vec2d(0., 0.)

        # Record num of steps
        self.num_steps = 0

        # Record whether the robot hit something
        self.hit = 0

        # Add borders in the space
        self.add_borders();

        # Add the robot in the space.
        self.r_width = 80
        self.r_height = 60
        self.add_robot(100, 100)

        # Add some obstacles in the space
        # For now, the obstacles are static
        self.num_obstacles = 10
        self.obstacles = []
        for i in range(self.num_obstacles):
            self.obstacles.append(self.add_obstacle(random.randint(30, width-30), random.randint(30, height-30)))

        # Add the goal in the space
        self.add_goal(width - 100, 100)

        # Draw stuffs on the screen
        self.draw_options = pymunk.pygame_util.DrawOptions(screen)

    def add_borders(self):
        borders = [
            pymunk.Segment(
                self.space.static_body,
                (0, 1), (0, height), 5),
            pymunk.Segment(
                self.space.static_body,
                (1, height), (width, height), 5),
            pymunk.Segment(
                self.space.static_body,
                (width - 1, height), (width - 1, 1), 5),
            pymunk.Segment(
                self.space.static_body,
                (1, 1), (width, 1), 5)  # ,
            # pymunk.Segment(
            #     self.space.static_body,
            #     (width/2., 1), (width/2., height*0.75), 5)
        ]
        for b in borders:
            b.friction = 1.
            b.group = 1
            b.collision_type = 1
            b.color = THECOLORS['brown']
            b.elasticity = 1
        self.space.add(borders)

    def add_robot(self, x, y):
        """Add a circle robot at a given position"""
        mass = 1
        radius = robot_radius
        inertia = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
        self.robot_body = pymunk.Body(mass, inertia)
        self.robot_body.position = x, y
        self.robot_body.velocity = random.choice([(1, 10), (-1, 10)])
        self.robot_body.velocity_func = self.constant_velocity

        self.robot_shape = pymunk.Circle(self.robot_body, radius, (0, 0))
        self.robot_shape.color = THECOLORS["orange"]
        self.robot_shape.elasticity = 1
        self.robot_shape.collision_type = 2
        self.space.add(self.robot_body, self.robot_shape)

    def add_obstacle(self, x, y):
        """Add an obstacle at a given position"""
        # mass = 1
        radius = obs_radius
        # inertia = pymunk.moment_for_circle(mass, 0, radius, (0,0))
        # obs_body = pymunk.Body(mass, inertia)

        obs_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        obs_body.position = x, y
        obs_shape = pymunk.Circle(obs_body, radius, (0, 0))
        obs_shape.color = THECOLORS["blue"]
        obs_shape.elasticity = 1
        self.space.add(obs_body, obs_shape)
        return obs_shape

    def add_goal(self, x, y):
        """Add the goal at a given position"""
        self.goal = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.goal.position = x, y
        self.goal_radius = 10
        self.goal_shape = pymunk.Circle(self.goal, self.goal_radius, (0, 0))
        self.goal_shape.color = THECOLORS["red"]
        self.goal_shape.collision_type = 2
        self.goal_shape.elasticity = 0
        self.space.add(self.goal, self.goal_shape)

    # Keep robot velocity at a static value
    def constant_velocity(self,body, gravity, damping, dt):
        body.velocity = body.velocity.normalized() * robot_velocity

    def get_sensor_data(self):
        readings = []
        for o in self.obstacles:
            distance = self.robot_body.position.get_distance(o.body.position)
            if distance <= sensor_range + obs_radius:
                readings.append(distance-robot_radius-obs_radius)
        return readings
    def check_reach_goal(self):
        distance2goal = self.robot_body.position.get_distance(self.goal.position)
        if distance2goal <= robot_radius + self.goal_radius:
            return True
        else:
            return False

    def get_reward(self,readings):
        reward = 0
        distance2goal = self.robot_body.position.get_distance(self.goal.position)
        if self.check_reach_goal():
            reward += 1000
        reward -= int(distance2goal/math.sqrt(width**2 + height**2) * 400)
        if readings != []:
            for d in readings:
                reward += int(200/(sensor_range-robot_radius) * d - 200)
        return reward

    def frame_step(self):
        # Update the screen and stuff.
        screen.fill(THECOLORS["white"])
        self.update(10)
        self.num_steps += 1

        # Align the robot's pointing angle to its velocity
        r_v = self.robot_body.velocity.get_length()
        r_vx = self.robot_body.velocity.x
        r_vy = self.robot_body.velocity.y
        if r_v != 0:
            if r_vx >= 0:
                self.robot_body.angle = math.asin(r_vy/r_v)
            else:
                self.robot_body.angle = math.pi - math.asin(r_vy / r_v)

        # Manually control the robot's action
        for event in pygame.event.get():
            if event.type == KEYDOWN and event.key == K_RIGHT:
                self.robot_body.velocity = self.robot_body.velocity.rotated_degrees(-15)
            elif event.type == KEYDOWN and event.key == K_LEFT:
                self.robot_body.velocity = self.robot_body.velocity.rotated_degrees(15)

        # Stop if reach the goal
        if self.check_reach_goal():
            self.robot_body.velocity = 0,0

        # Get the current state of the robot
        position = self.robot_body.position
        readings = self.get_sensor_data()
        state = {
            "position": position,
            "readings": readings
        }
        reward = self.get_reward(readings)

        return state, reward

    def update(self, freq):
        self.space.debug_draw(self.draw_options)
        self.space.step(1 / freq)
        pygame.display.flip()
        clock.tick(freq)


if __name__ == "__main__":
    game_class = GameClass()

    # Exit the game
    for event in pygame.event.get():
        if event.type == QUIT:
            sys.exit(0)
        elif event.type == KEYDOWN and (event.key in [K_ESCAPE, K_q]):
            sys.exit(0)

    # Game loop
    while True:
        state, reward = game_class.frame_step()
        print(reward)