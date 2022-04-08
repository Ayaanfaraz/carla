#!/usr/bin/env python3
import glob
import os
import sys
import random
import time
import sys
import numpy as np
import cv2
import math

from collections import deque
import tensorflow as tf
from threading import Thread
from tqdm import tqdm

import matplotlib.pyplot as plt
import xception
import torch.nn as nn
import torch.optim as optim
import torch

import jerrys_helpers
import tiramisuModel.tiramisu as tiramisu
from torchvision import transforms
from mergedModel import MyEnsemble as fusionModel

"Starting script for any carla programming"

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from carla import ColorConverter as cc

try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import queue
except ImportError:
    import Queue as queue

model = fusionModel(
                semantic_model=xception.xception(num_classes=2048, pretrained=False).float(),
                uncertainty_model=xception.xception(num_classes=2048, pretrained=False).float()).to(device='cuda:1')

semantic_uncertainty_model = tiramisu.FCDenseNet67(n_classes=23).to(device='cuda:0')
semantic_uncertainty_model.float()
jerrys_helpers.load_weights(semantic_uncertainty_model,'models/weights67latest.th')

transform_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.534603774547577, 0.570066750049591, 0.589080333709717],
    std = [0.186295211315155, 0.181921467185020, 0.196240469813347])
])

def process_img(image):
        # #image.convert(cc.CityScapesPalette)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        image = array[:, :, :3]
       
        ## #Get semantic image ######
        # Normalize rgb input Image
        normalized_image = transform_norm(image)
        rgb_input = torch.unsqueeze(normalized_image, 0)
        rgb_input = rgb_input.to(torch.device("cuda:0"))
        # Get semantic segmented raw output
        semantic_uncertainty_model.eval().to(device='cuda:0')
        model_output = semantic_uncertainty_model(rgb_input) #Put single image rgb in tensor and pass in
        raw_semantic = jerrys_helpers.get_predictions(model_output) #Gets an unlabeled semantic image (red one)
        rgb_semantic = jerrys_helpers.color_semantic(raw_semantic[0]) #gets color converted semantic (like our convert cityscape)
        #Convert Jerry model float64 input to uint8
        rgb_semantic = cv2.cuda.normalize(src=rgb_semantic, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        ##### Get Semantic Image #######


        #### Get Uncertainty Image ######
        semantic_uncertainty_model.train().to(device='cuda:0')
        mc_results = []
        output = semantic_uncertainty_model(rgb_input).detach().cpu().numpy()
        output = np.squeeze(output)
        # RESHAPE OUTPUT BEFORE PUTTING IT INTO mc_results
        # reshape into (480000, 23)
        # then softmax it
        output = jerrys_helpers.get_pixels(output)
        output = jerrys_helpers.softmax(output)
        mc_results.append(output)
        
        # boom we got num_samples passes of a single img thru the NN
        # now we use those samples to make uncertainty maps  
        mc_results = [mc_results]
        aleatoric = jerrys_helpers.calc_aleatoric(mc_results)[0]
        aleatoric = np.reshape(aleatoric, (300, 300))
        aleatoric = cv2.cuda.normalize(src=aleatoric, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        aleatoric = cv2.merge((aleatoric,aleatoric,aleatoric))
        ###### Get Uncertainty Image ######
        # cv2.imshow("semantic cv2", rgb_semantic)
        # cv2.imshow("Aleatoric cv2", aleatoric)
        return rgb_semantic, aleatoric
        

class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    return array
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False


def main():
    actor_list = []
    pygame.init()

    # display = pygame.display.set_mode(
    #     (800, 600),
    #     pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()

    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(2.0)

    world = client.get_world()

    try:
        m = world.get_map()
        start_pose = random.choice(m.get_spawn_points())
        waypoint = m.get_waypoint(start_pose.location)

        blueprint_library = world.get_blueprint_library()

        vehicle = world.spawn_actor(
            random.choice(blueprint_library.filter('vehicle.*')),
            start_pose)
        actor_list.append(vehicle)
        #vehicle.set_simulate_physics(False)

        camera = blueprint_library.find('sensor.camera.rgb')
        camera.set_attribute("image_size_x", f"{300}")
        camera.set_attribute("image_size_y", f"{300}")

        camera_rgb = world.spawn_actor(
            camera,
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            attach_to=vehicle)
        actor_list.append(camera_rgb)

        camera_semseg = world.spawn_actor(
            blueprint_library.find('sensor.camera.semantic_segmentation'),
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            attach_to=vehicle)
        actor_list.append(camera_semseg)

        # Create a synchronous mode context.
        with CarlaSyncMode(world, camera_rgb, camera_semseg, fps=30) as sync_mode:
            while True:
                if should_quit():
                    return
                #clock.tick()

                # Advance the simulation and wait for the data.
                snapshot, image_rgb, image_semseg = sync_mode.tick(timeout=2.0)

                # Choose the next waypoint and update the car location.
                waypoint = random.choice(waypoint.next(1.5))
                #vehicle.set_transform(waypoint.transform)
                vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer= 0.0 ))

                image_semseg.convert(carla.ColorConverter.CityScapesPalette)
                fps = round(1.0 / snapshot.timestamp.delta_seconds)

                cv2.imshow("cv2",draw_image(fps, image_rgb))
                start = time.time()
                rgb_semantic, aleatoric = process_img(image_rgb)
                process_time = time.time()-start
                print("process semantic and uncertainty: ",round(process_time,2) )
                model_start = time.time()
                
                uncertainty_tensor = (torch.unsqueeze(torch.from_numpy(aleatoric), 0).permute(0,3,1,2)/255).to(device='cuda:1')
                semantic_tensor = (torch.unsqueeze(torch.from_numpy(rgb_semantic), 0).permute(0,3,1,2)/255).to(device='cuda:1')
                action = torch.argmax(model( semantic_tensor, uncertainty_tensor)[0])
                
                model_end = time.time() - model_start
                print("model process time: ", (round(model_end,2)) )
                if action == 0:
                    vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer= 0.0 ))
                if action == 1:
                    vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1.0))
                if action == 2:
                    vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1.0))
                # Draw the display.
                # draw_image(display, image_rgb)
                # draw_image(display, image_semseg, blend=True)
                # display.blit(
                #     font.render('% 5d FPS (real)' % clock.get_fps(), True, (255, 255, 255)),
                #     (8, 10))
                # display.blit(
                #     font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),
                #     (8, 28))
                # pygame.display.flip()

    finally:

        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()

        pygame.quit()
        print('done.')


if __name__ == '__main__':

    try:

        main()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
