#!/usr/bin/env python3
import glob
import os
import sys
import random
import time
import sys
from matplotlib import image
import numpy as np
import cv2
import math

from collections import deque
# import pygame
try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')
import tensorflow as tf
from threading import Thread
from multiprocessing import Process
from tqdm import tqdm
import queue

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

IM_WIDTH = 300
IM_HEIGHT = 300
SECONDS_PER_EPISODE = 20
REPLAY_MEMORY_SIZE = 5_000

MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
MODEL_NAME = "Xception"

MEMORY_FRACTION = 0.8
MIN_REWARD = -200

EPISODES = 100

DISCOUNT = 0.99
epsilon = 1
EPSILON_DECAY = 0.95 ## 0.9975 99975
MIN_EPSILON = 0.001

AGGREGATE_STATS_EVERY = 5  ## checking per 5 episodes
SHOW_PREVIEW  = True    ## for debugging purpose

#Load up Jerrys pretrained model
semantic_uncertainty_model = tiramisu.FCDenseNet67(n_classes=23).to(device='cuda:0')
semantic_uncertainty_model.float()
jerrys_helpers.load_weights(semantic_uncertainty_model,'models/weights67latest.th')

transform_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.534603774547577, 0.570066750049591, 0.589080333709717],
    std = [0.186295211315155, 0.181921467185020, 0.196240469813347])
])

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
        try:
            data = [self._retrieve_data(q, timeout) for q in self._queues]
            assert all(x.frame == self.frame for x in data)
            return data
        except:
            time.sleep(10)
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


class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0   ## full turn for every single time
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    num_timesteps = 0

    def __init__(self):
        self.client = carla.Client('127.0.0.1', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world('Town04')
        self.map = self.world.get_map()   ## added for map creating
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]  ## grab tesla model3 from library

    def reset(self):
        self.collision_hist = []    
        self.actor_list = []
        self.num_timesteps = 1
        
        self.waypoints = self.client.get_world().get_map().generate_waypoints(distance=5.0)
        self.spawn_point = self.waypoints[0].transform

        self.spawn_point.location.z += 2
        self.vehicle = self.world.spawn_actor(self.model_3, self.spawn_point)  ## changed for adding waypoints

        self.actor_list.append(self.vehicle)

        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.rgb_cam.set_attribute("fov", f"110")  ## fov, field of view

        self.ss_cam = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        self.ss_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.ss_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.ss_cam.set_attribute("fov", f"110")

        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.rgb_sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.rgb_sensor)
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0)) # initially passing some commands seems to help with time. Not sure why.
        #time.sleep(4)  # sleep to get things started and to not detect a collision when the car spawns/falls from sky.
        #We need to add trhis after we start the synchronous mode in the main loop
        
        colsensor = self.world.get_blueprint_library().find('sensor.other.collision')
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.colsensor.listen(lambda event: self.collision_data(event))
        self.actor_list.append(self.colsensor)

    def collision_data(self, event):
        self.collision_hist.append(event)

    def process_img(self, image):
        #image.convert(cc.CityScapesPalette)
        i = np.array(image.raw_data, dtype=np.dtype("uint8"))
        i2 = i.reshape((self.im_height, self.im_width, 4))
        image = i2[:, :, :3]
       
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
        rgb_semantic = cv2.normalize(src=rgb_semantic, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
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
        aleatoric = np.reshape(aleatoric, (IM_HEIGHT, IM_WIDTH))
        aleatoric = cv2.normalize(src=aleatoric, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        aleatoric = cv2.merge((aleatoric,aleatoric,aleatoric))
        # cv2.imshow("Semantic Segmentation", rgb_semantic)
        # cv2.imshow("Aleatoric Uncertainty", al)
        return rgb_semantic, aleatoric
    
    def draw_image(self, image):
        array = np.array(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        return array


    def step(self, action, sync_mode):
        '''
        For now let's just pass steer left, straight, right
        0, 1, 2
        '''
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer= 0.0 ))
        if action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1.0*self.STEER_AMT))
        if action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1.0*self.STEER_AMT))
        
        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
      
        car_location = carla.Actor.get_location(self.actor_list[0])
        nearest_waypoint = self.client.get_world().get_map().get_waypoint(car_location,project_to_road=False, lane_type=(carla.LaneType.Driving))

        if len(self.collision_hist) != 0:
            done = True
            reward = -300
        elif kmh < 30:
            done = False
            reward = -5
        elif nearest_waypoint is not None and carla.Location.distance(car_location, nearest_waypoint.transform.location) <= 0.1:
            done = False
            reward = 25
        else:
            done = False
            reward = 30

        if self.num_timesteps > 300:  ## when to stop
            done = True

        try:
            snapshot, image_rgb = sync_mode.tick(timeout=20.0) #This is resets tick
        except:
            print("error")
            return None, reward, done, True
        proc_start = time.time()
        semantic_segmentation, aleatoric_uncertainty = env.process_img(image_rgb)
        proc_end = time.time()-proc_start
        # print("Process duration: ", proc_end)
        image_rgbs = env.draw_image(image_rgb)
        cv2.imshow("Ground Truth RGB",image_rgbs)
        cv2.waitKey(1)
        cv2.imshow("Semantic Segmentation", semantic_segmentation)
        cv2.waitKey(1)
        cv2.imshow("Aleatoric Uncertainty", aleatoric_uncertainty)
        cv2.waitKey(1)

        # cv2.imwrite(("image/"+str(self.num_timesteps)+"_rgb.png"), image_rgbs)
        # cv2.imwrite(("image/"+str(self.num_timesteps)+"_sem.png"), semantic_segmentation)
        # cv2.imwrite(("image/"+str(self.num_timesteps)+"_unc.png"), aleatoric_uncertainty)
        current_state = (semantic_segmentation, aleatoric_uncertainty)

        return current_state, reward, done, None



class DQNAgent:
    def __init__(self):


        ## replay_memory is used to remember the sized previous actions, and then fit our model of this amout of memory by doing random sampling
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)   ## batch step
        self.target_update_counter = 0  # will track when it's time to update the target model
       
        self.model = fusionModel(semantic_model=self.create_model(), uncertainty_model=self.create_model()).to(device='cuda:1')
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "model" in name:
                    param.requires_grad = False
                    print(name)
        # print("\n\n\n")
        # for param in self.model.parameters():
        #     if param.requires_grad:
        #         print(param)

        #self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.001)
        self.terminate = False  # Should we quit?
        self.training_initialized = False  # waiting for TF to get rolling

    def create_model(self):
        model = xception.xception(num_classes=2048, pretrained=False).to(device="cuda:1")
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)= (current_state, action, reward, new_state, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)


    def train(self):
        ## starting training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        ## if we do have the proper amount of data to train, we need to randomly select the data we want to train off from our memory
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        torch.cuda.empty_cache()
        states, targets_f = [],[]
        torch.autograd.set_detect_anomaly(True)
        criterion = nn.MSELoss().to(device='cuda:1')
        for state, action, reward, next_state, done in minibatch:

            #Belman Equation future rewards calculated 
            target = reward
            uncertainty_tensor = (torch.unsqueeze(torch.from_numpy(next_state[1]), 0).permute(0,3,1,2)/255).to(device='cuda:1')
            semantic_tensor = (torch.unsqueeze(torch.from_numpy(next_state[0]), 0).permute(0,3,1,2)/255).to(device='cuda:1')
            # if not done, predict future discounted reward with the Bellman equation
            if not done:
                target = (reward + DISCOUNT * torch.amax(
                    self.model(semantic_tensor, uncertainty_tensor)[0]).item()) #cuda 1 here
            
            uncertainty_target = (torch.unsqueeze(torch.from_numpy(state[1]), 0).permute(0,3,1,2)/255).to(device='cuda:1')
            semantic_target = (torch.unsqueeze(torch.from_numpy(state[0]), 0).permute(0,3,1,2)/255).to(device='cuda:1')
            target_f = self.model(semantic_target, uncertainty_target) # cuda 1 here
            target_f[0][action] = target
            target_f = target_f.detach()
            # filtering out states and targets for training
            states.append((semantic_target,uncertainty_target)) #Add the uncertainty/semantic segmented tuple
            targets_f.append(target_f)


        self.model.train().to(device='cuda:1')
        #del self.loss
        for i in range(len(states)):
            self.optimizer.zero_grad()

            x1    = states[i][0]#Semantic
            x2    = states[i][1]#Uncertainty
            y     = targets_f[i]
            #print("target y size: ", y.shape)

            yhat = self.model(x1,x2)
           ## print("yhat: ", yhat.shape)
            loss=criterion(yhat, y)

            loss.backward()
            self.optimizer.step()
        print("training complete for one batch")

    def get_qs(self, state):
        torch.cuda.empty_cache()
        uncertainty_tensor = (torch.unsqueeze(torch.from_numpy(state[1]), 0).permute(0,3,1,2)/255).to(device='cuda:1')
        semantic_tensor = (torch.unsqueeze(torch.from_numpy(state[0]), 0).permute(0,3,1,2)/255).to(device='cuda:1')
        q_vector = self.model(semantic_tensor, uncertainty_tensor)[0]
        #print("q vector: ", q_vector.item())
        return q_vector
        
    def train_in_loop(self):
        self.training_initialized = True

        while True:
            if self.terminate:
                return
            self.train()
            #time.sleep(0.4)

if __name__ == '__main__':
    FPS = 20
    # For stats
    ep_rewards = [-200]
    process_start = time.time()
    # For more repetitive results
    #random.seed(1)
    np.random.seed(2021)

    # Create models folder, this is where the model will go 
    if not os.path.isdir('models'):
        os.makedirs('models')

    # Create agent and environment
    agent = DQNAgent()
    env = CarEnv()

    # Start training thread and wait for training to be initialized
    trainer_thread = Process(target=agent.train_in_loop, daemon=True)
    trainer_thread.start()
    # while not agent.training_initialized:
    #     time.sleep(0.1)

    rewards = []
    episode_list = []
    # Iterate over episodes
    for episode in tqdm(range(1, EPISODES + 1), unit='episodes'):
        #try:
            pygame.init()
            env.collision_hist = []
            # Restarting episode - reset episode reward and step number
            episode_reward = 0
            step = 1

            # Reset environment and sensors etc
            env.reset()
            current_state = None
            # Reset done flag
            done = False
            episode_start = time.time()
            # Play for given number of seconds only
            #with synchronus mode()
            #state = rgb
            with CarlaSyncMode(env.world, env.rgb_sensor, fps=30) as sync_mode:
                
                #add initial delay to speed up car spawn
                for i in range(50):
                    try:
                        snapshot, image_rgb = sync_mode.tick(timeout=20.0)
                    except:
                        print("error")
                        break
                
                for i in range(25):
                    try:
                        snapshot, image_rgb = sync_mode.tick(timeout=20.0)
                        env.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer= 0.0 ))
                    except:
                        print("error")
                        break

                try:
                    snapshot, image_rgb = sync_mode.tick(timeout=20.0) #This is resets tick
                except:
                    print("error")
                    break
                proc_start = time.time()
                semantic_segmentation, aleatoric_uncertainty = env.process_img(image_rgb)
                proc_end = time.time()-proc_start
                #print("Process duration: ", proc_end)
                # cv2.imshow("Ground Truth RGB", env.draw_image(image_rgb))
                # cv2.imshow("Semantic Segmentation", semantic_segmentation)
                # cv2.imshow("Aleatoric Uncertainty", aleatoric_uncertainty)
                # cv2.imwrite(("image1/RGB"+str(env.num_timesteps)+".png"), image_rgb)
                # cv2.imwrite(("image1/Semantic"+str(env.num_timesteps)+".png"), semantic_segmentation)
                # cv2.imwrite(("image1/Aleatoric"+str(env.num_timesteps)+".png"), aleatoric_uncertainty)
                current_state = (semantic_segmentation, aleatoric_uncertainty)
                
                # for event in pygame.event.get():
                #     pass

                while True:
                    #Visualizations#

                    if np.random.random() > epsilon:
                        # Get action from Q table
                        start = time.time()
                        action = torch.argmax(agent.get_qs(current_state)).item()
                        #print("model time: ", time.time()-start)
                    else:
                        # Get random action
                        action = np.random.randint(0, 3)
                        # This takes no time, so we add a delay matching 60 FPS (prediction above takes longer)
                        #time.sleep(0.04)

                    new_state, reward, done, err = env.step(action, sync_mode)

                    if err:
                        break

                    # Transform new continous state to new discrete state and count reward
                    episode_reward += reward

                    # Every step we update replay memory
                    agent.update_replay_memory((current_state, action, reward, new_state, done))

                    current_state = new_state
                    step += 1
                    print("step: "+ str(step)+ " reward: "+str(reward)+" action: ", str(action))
                    env.num_timesteps = step

                    if done:
                        break

            # minibatch = random.sample(agent.replay_memory, MINIBATCH_SIZE)
            # for state, action, reward, next_state, done in minibatch:
            #     print("state size: ", state[0].shape)
            #     print("next state size: ", next_state[1].shape)

            episode_list.append(episode)
            rewards.append(episode_reward)
            # End of episode - destroy agents
            for actor in env.actor_list:
                actor.destroy()
            pygame.quit()

            # Append episode reward to a list and log stats (every given number of episodes)
            ep_rewards.append(episode_reward)

            # Decay epsilon
            if epsilon > MIN_EPSILON:
                epsilon *= EPSILON_DECAY
                epsilon = max(MIN_EPSILON, epsilon)
            
            #plt.figure(1)
            plt.xlabel('Episodes')
            plt.ylabel('Rewards')
            plt.title(str('Training'))  
            plt.plot(episode_list, rewards)
            plt.savefig('_out/reward_graph.png')

    print("Total Runtime for 100 Episodes: ", time.time()-process_start)
    # Set termination flag for training thread and wait for it to finish
    agent.terminate = True
    trainer_thread.join()
    #agent.model.save('models/all_waypoints_model')