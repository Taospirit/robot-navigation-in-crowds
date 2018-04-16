import sys, random
import math
from math import sin, cos
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
            self.obstacles.append(self.add_obstacle(random.randint(0, width), random.randint(0, height)))

        # Add the goal in the space
        self.draw_goal(width - 100, 100)

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

    # def add_robot(self, x, y):
    #     """Add a rectangle robot at a given position"""
    #     mass = 1
    #     vertices = [(self.r_width / 2, self.r_height / 2),
    #                 (-self.r_width / 2, self.r_height / 2),
    #                 (-self.r_width / 2, -self.r_height / 2),
    #                 (self.r_width / 2, -self.r_height / 2)]
    #     inertia = pymunk.moment_for_poly(mass, vertices, (0, 0))
    #     self.robot_body = pymunk.Body(mass, inertia)
    #     self.robot_body.position = x, y
    #     self.robot_shape = pymunk.Poly(self.robot_body, vertices)
    #     self.robot_shape.color = THECOLORS["orange"]
    #     self.space.add(self.robot_body, self.robot_shape)

    def add_robot(self, x, y):
        """Add a circle robot at a given position"""
        mass = 1
        radius = 30
        inertia = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
        self.robot_body = pymunk.Body(mass, inertia)
        self.robot_body.position = x, y
        self.robot_body.velocity = random.choice([(1, 10), (-1, 10)])
        self.robot_body.velocity_func = self.constant_velocity

        self.robot_shape = pymunk.Circle(self.robot_body, radius, (0, 0))
        self.robot_shape.color = THECOLORS["orange"]
        self.robot_shape.elasticity = 1
        self.space.add(self.robot_body, self.robot_shape)

    def add_obstacle(self, x, y):
        """Add an obstacle at a given position"""
        # mass = 1
        radius = 30
        # inertia = pymunk.moment_for_circle(mass, 0, radius, (0,0))
        # obs_body = pymunk.Body(mass, inertia)

        obs_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        obs_body.position = x, y
        obs_shape = pymunk.Circle(obs_body, radius, (0, 0))
        obs_shape.color = THECOLORS["blue"]
        obs_shape.elasticity = 1
        self.space.add(obs_body, obs_shape)
        return obs_shape

    def draw_goal(self, x, y):
        """Draw the goal at a given position"""
        goal = pymunk.Body(body_type=pymunk.Body.STATIC)
        goal.position = x, y
        goal_radius = 10
        goal_shape = pymunk.Circle(goal, goal_radius, (0, 0))
        goal_shape.color = THECOLORS["red"]
        self.space.add(goal, goal_shape)

    # def move_robot(self):
    #     """Align the velocity and robot angle"""
    #     # print("in align")
    #     speed = self.robot_body.velocity.get_length()
    #     direction = Vec2d(1., 0.).rotated(self.robot_body.angle)
    #     self.robot_body.velocity = speed * direction

    # Keep robot velocity at a static value
    def constant_velocity(self,body, gravity, damping, dt):
        # body.velocity = 200 * Vec2d(1., 0.).rotated(body.angle)
        body.velocity = body.velocity.normalized() * 200.


    # def random_move_robot(self):
    #     """Randomly move the robot"""
    #     print("in move")
    #     speed = 400
    #     angle = random.uniform(-1.5, 1.5)
    #     self.robot_body.angle += angle
    #     direction = Vec2d(1., 0.).rotated(self.robot_body.angle)
    #     self.robot_body.velocity = speed * direction
    #     print(math.degrees(angle))

    def get_sensor_data(self, x, y, angle):
        """Get the reading of a sensor, specify its positon on the robot and its pointing angle"""
        self.sensor = []
        x_r, y_r = self.robot_body.position
        sensor_range = 20
        sensor_resolution = 1
        sensor_points = []
        data = -1
        for d in range(1, sensor_range, sensor_resolution):
            x_p = x + d * cos(angle)
            y_p = y + d * sin(angle)
            x_p, y_p = self.robot_body.local_to_world([x_p, y_p])
            x_p, y_p = self.bound(x_p, y_p)
            sensor_point = [int(x_p), int(y_p)]
            obs = screen.get_at(sensor_point)
            sensor_points.append(sensor_point)
            if obs != THECOLORS["white"] and obs != THECOLORS["yellow"] and obs != THECOLORS["darkred"]:
                data = d
                break

        """Draw the sensor on the screen"""
        # sensor_draw_points = [(sensor_points[0][0], height - sensor_points[0][1]),
        #                       (sensor_points[-1][0], height - sensor_points[-1][1])]
        # pygame.draw.line(screen, THECOLORS["darkred"], sensor_draw_points[0], sensor_draw_points[1], 5)

        self.sensor.append(sensor_points)
        return data

    def get_all_sensors_data(self):
        datas = []
        datas.append(self.get_sensor_data(self.r_width / 2, 0, 0))
        datas.append(self.get_sensor_data(self.r_width / 2, self.r_height / 2, math.radians(45)))
        datas.append(self.get_sensor_data(self.r_width / 2, -self.r_height / 2, math.radians(-45)))
        datas.append(self.get_sensor_data(-self.r_width / 2, self.r_height / 2, math.radians(135)))
        datas.append(self.get_sensor_data(-self.r_width / 2, 0, math.radians(180)))
        datas.append(self.get_sensor_data(-self.r_width / 2, -self.r_height / 2, math.radians(-135)))
        # pygame.display.update()
        for data in datas:
            if data > 0:
                print("Sensor datas:")
                print(datas)
        return datas

    def bound(self, x_p, y_p):
        """Make sure the point is in the screen"""
        if x_p < 0:
            x_p = 0
        elif x_p > width:
            x_p = width
        if y_p < 0:
            y_p = 0
        elif y_p > height:
            y_p = height
        return x_p, y_p

    def check_hit(self, datas):
        """Check if the robot hit something"""
        for data in datas:
            if data > 0:
                return 1
        return 0

    def recover_from_hit(self):
        while self.hit:
            print("in recover")
            # Go backwards.
            direction = Vec2d(1., 0.).rotated(self.robot_body.angle)
            self.robot_body.velocity = -100 * direction
            for i in range(10):
                screen.fill(THECOLORS["yellow"])
                self.update(10)
            self.robot_body.angle += .5  # Turn a little.
            self.move_robot()
            # Check hit
            self.hit = self.check_hit(self.get_all_sensors_data())

    def frame_step(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                sys.exit(0)
            elif event.type == KEYDOWN and (event.key in [K_ESCAPE, K_q]):
                sys.exit(0)
        # print("in frame step")
        # print(self.num_steps)
        # Update the screen and stuff.
        screen.fill(THECOLORS["white"])
        self.update(10)
        self.num_steps += 1

        # # Move the robot at every 100 ms.
        # if self.num_steps % 100 == 0:
        #     self.random_move_robot()
        # else:
        #     self.move_robot()

        # Align the robot's pointing angle to its velocity
        r_v = self.robot_body.velocity.get_length()
        r_vx = self.robot_body.velocity.x
        r_vy = self.robot_body.velocity.y
        if r_vx >= 0:
            self.robot_body.angle = math.asin(r_vy/r_v)
        else:
            self.robot_body.angle = math.pi - math.asin(r_vy / r_v)



        for event in pygame.event.get():
            # if event.type == KEYDOWN and event.key == K_UP:
            #     self.robot_body.velocity += 40 * Vec2d(1., 0.).rotated(self.robot_body.angle)
            # # elif event.type == KEYUP and event.key == K_UP:
            # #     self.robot_body.velocity = 0, 0
            #
            # elif event.type == KEYDOWN and event.key == K_DOWN:
            #     self.robot_body.velocity += -40 * Vec2d(1., 0.).rotated(self.robot_body.angle)
            # # elif event.type == KEYUP and event.key == K_DOWN:
            # #     self.robot_body.velocity = 0, 0
            # elif event.type == KEYDOWN and event.key == K_SPACE:
            #     self.robot_body.velocity = 0, 0

            if event.type == KEYDOWN and event.key == K_RIGHT:
                self.robot_body.velocity = self.robot_body.velocity.rotated_degrees(-15)
            # elif event.type == KEYUP and event.key == K_RIGHT:
            #     self.robot_body.velocity -= 0

            elif event.type == KEYDOWN and event.key == K_LEFT:
                # self.robot_body.angle += .2
                self.robot_body.velocity = self.robot_body.velocity.rotated_degrees(15)
            # elif event.type == KEYUP and event.key == K_LEFT:
            #     self.robot_body.angle += 0


        # # If hit, let the robot make reaction
        # self.hit = self.check_hit(self.get_all_sensors_data())
        # if self.hit == 1:
        #     self.recover_from_hit()

    def update(self, freq):

        self.space.debug_draw(self.draw_options)
        self.space.step(1 / freq)
        pygame.display.flip()
        clock.tick(freq)


if __name__ == "__main__":
    game_class = GameClass()

    while True:
        game_class.frame_step()
