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
import pickle

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
TIMESTEPS_PER_EPISODE = 400
REPLAY_MEMORY_SIZE = 10_000

MIN_REPLAY_MEMORY_SIZE = 1000
MINIBATCH_SIZE = 64
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4

EPISODES = 10_000

DISCOUNT = 0.99 # Maybe change to 0.999?
epsilon = 1
EPSILON_DECAY = 0.9991#0.9659 ## 0.9975 99975
MIN_EPSILON = 0.001
Var = 0

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
        self.obstacle_data=[]    
        self.actor_list = []
        self.num_timesteps = 1
        
        try:
            self.waypoints = self.client.get_world().get_map().generate_waypoints(distance=3.0)
        except:
            time.sleep(10)
            self.waypoints = self.client.get_world().get_map().generate_waypoints(distance=3.0)

        # for i in range(len(self.waypoints)):
        #             self.world.debug.draw_string(self.waypoints[i].transform.location, 'O', draw_shadow=False,
        #                            color=carla.Color(r=0, g=255, b=0), life_time=40,
        #                            persistent_lines=True)

        self.spawn_point = self.waypoints[0].transform #random.choice(self.waypoints).transform #Used to be waypoint[0]
        self.initial_waypoint = self.waypoints[0]
        self.spawn_point.location.z += 2
        self.vehicle = self.world.spawn_actor(self.model_3, self.spawn_point)  ## changed for adding waypoints

        obstacle_location = self.spawn_point
        obstacle_location.location.x -= 38
        #print (self.world.get_blueprint_library())
        self.obstacle = self.world.spawn_actor(self.world.get_blueprint_library().filter('bmw')[0], obstacle_location)

        self.actor_list.append(self.vehicle)
        self.actor_list.append(self.obstacle)

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
        
        self.sem_sensor = self.world.spawn_actor(self.ss_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sem_sensor)
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0)) # initially passing some commands seems to help with time. Not sure why.
        #time.sleep(4)  # sleep to get things started and to not detect a collision when the car spawns/falls from sky.
        #We need to add trhis after we start the synchronous mode in the main loop
        
        colsensor = self.world.get_blueprint_library().find('sensor.other.collision')
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.colsensor.listen(lambda event: self.collision_data(event))
        self.actor_list.append(self.colsensor)

        obstacle_detector = self.world.get_blueprint_library().find('sensor.other.obstacle')
        obstacle_detector.set_attribute("distance", f"17")
        self.obstacle_detector = self.world.spawn_actor(obstacle_detector, transform, attach_to=self.vehicle)
        self.obstacle_detector.listen(lambda event: self.obstacle_hist(event))
        self.actor_list.append(self.obstacle_detector)

    def collision_data(self, event):
        self.collision_hist.append(event)
    
    def obstacle_hist(self, event):
        self.obstacle_data.append(event)

    def process_sem(self, image):
        image.convert(cc.CityScapesPalette)
        i = np.array(image.raw_data)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        return i3

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

        # cv2.imshow("Semantic Segmentation", rgb_semantic)
        # cv2.imshow("Aleatoric Uncertainty", al)
        return rgb_semantic
    
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
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer= 0.0 ))
        if action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=1.0*self.STEER_AMT))
        if action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=-1.0*self.STEER_AMT))
        
        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
      
        car_location = carla.Actor.get_location(self.actor_list[0])
        # nearest_waypoint = self.client.get_world().get_map().get_waypoint(car_location,project_to_road=True, lane_type=(carla.LaneType.Driving))#self.initial_waypoint.next(3.0)[0]#
        # dist = None
        # if nearest_waypoint is not None:
        # #     self.world.debug.draw_string(nearest_waypoint.transform.location, 'O', draw_shadow=False,
        # #                             color=carla.Color(r=0, g=255, b=0), life_time=40,
        # #                             persistent_lines=True)
                                    
        #     dist = round(carla.Location.distance(car_location, nearest_waypoint.transform.location),2)
        reward = 0
        if len(self.collision_hist) != 0:
            done = True
            reward = -100
        elif len(self.obstacle_data)!=0 :
            reward = -15+self.obstacle_data[-1].distance
            done = False
        elif kmh < 30:
            done = False
            reward = -5
        # elif nearest_waypoint is not None and dist <= 0.4:
        #     done = False
        #     reward = 25
        else:
            done = False
            reward = 5
        if self.num_timesteps > TIMESTEPS_PER_EPISODE:  ## when to stop
            done = True
        
        try:
            snapshot, image_rgb, image_semantic = sync_mode.tick(timeout=20.0) #This is next state image
        except:
            print("error")
            return None, reward, done, True #return reward for old state and next state image

        proc_start = time.time()
        semantic_segmentation = env.process_sem(image_semantic)
        proc_end = time.time()-proc_start
        # print("Process duration: ", proc_end)
        image_rgbs = env.draw_image(image_rgb)
        cv2.imshow("Ground Truth RGB",image_rgbs)
        cv2.waitKey(1)
        cv2.imshow("Semantic Segmentation", semantic_segmentation)
        cv2.waitKey(1)

        current_state = semantic_segmentation

        return current_state, reward, done, None



class DQNAgent:
    def __init__(self, loaded_state):


        ## replay_memory is used to remember the sized previous actions, and then fit our model of this amout of memory by doing random sampling
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)   ## batch step
        try:
            with open('_out/replay_memory.pkl','rb') as f:
                self.replay_memory = pickle.load(f)
        except:
            print("replay load error")
            pass

        self.target_update_counter = 0  # will track when it's time to update the target model
       
        self.model = self.create_model() # Single xception model

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                #print(name)
                if not "last_linear" in name:
                    param.requires_grad = False
                    #print(name)

        #self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.001)
        self.terminate = False  # Should we quit?
        self.training_initialized = False  # waiting for TF to get rolling

        try:
            self.model.load_state_dict(loaded_state['model_state_dict'])
            self.optimizer.load_state_dict(loaded_state['optimizer'])
        except:
            print("Model state error")
            pass


    def create_model(self):
        return xception.xception(num_classes=3, pretrained=False).to(device="cuda:1")
        #return model

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
        uncertainty_states = []
        semantic_states = []
        for state, action, reward, next_state, done in minibatch:

            #Belman Equation future rewards calculated 
            target = reward
            semantic_tensor = (torch.unsqueeze(torch.from_numpy(next_state), 0).permute(0,3,1,2)/255).to(device='cuda:1')
            # if not done, predict future discounted reward with the Bellman equation
            if not done:
                target = (reward + DISCOUNT * torch.amax(
                    self.model(semantic_tensor)[0]).item()) #cuda 1 here
                del semantic_tensor
            
            semantic_target = (torch.unsqueeze(torch.from_numpy(state), 0).permute(0,3,1,2)/255).to(device='cuda:1')
            # size is 1, 300, 300, 3
            # size 4, 300, 300, 3
            target_f = self.model(semantic_target) # cuda 1 here torch.tensor([0.9, 0.7, 0.5])
            del semantic_target

            target_f[0][action] = target
            target_f = target_f.detach()
            # filtering out states and targets for training

            semantic_state = (torch.from_numpy(state).permute(2,0,1)/255).to(device='cuda:1')

            semantic_states.append(semantic_state) #Add the uncertainty/semantic segmented tuple
            del semantic_state

            targets_f.append(torch.squeeze(target_f)) # 16 1 by 3 tensors (list of q value outputs)
        # print("targets_f: ", targets_f)
        
        self.model.train().to(device='cuda:1')
        #del self.loss
        for i in range(TRAINING_BATCH_SIZE):
            self.optimizer.zero_grad()
            x1    = torch.stack((semantic_states[i:i+TRAINING_BATCH_SIZE]))#Semantic -> [[3x4]] -> 1x3x4 -> #

            y     = torch.stack(targets_f[i:i+TRAINING_BATCH_SIZE]) #batch size of 4 labels
            # print("y shape is:", y.shape)
            yhat = self.model(x1)
            # print("yhat is: ",yhat.shape) #4,3 tensor
            
            loss=criterion(yhat, y)
            loss.backward()
            self.optimizer.step()

            i+=TRAINING_BATCH_SIZE
        # print("training complete for one batch")

    def get_qs(self, state):
        torch.cuda.empty_cache()
        with torch.no_grad():
            return self.model(
                (torch.unsqueeze(torch.from_numpy(state), 0).permute(0,3,1,2)/255).to(device='cuda:1'))[0]
            #print("q vector: ", q_vector.item())
            #return q_vector
        
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
    random.seed(1)
    #np.random.seed(2021)

    # Create models folder, this is where the model will go 
    if not os.path.isdir('models'):
        os.makedirs('models')

    startEpisode = 1
    loaded_state = None
    try:
        loaded_state = torch.load('models/saved_model.pt')
        epsilon = loaded_state['epsilon']
        startEpisode = loaded_state['episode']
        print("Start at Episode: ", startEpisode)
    except:
        pass
    # Create agent and environment
    agent = DQNAgent(loaded_state)
    env = CarEnv()

    # Start training thread and wait for training to be initialized
    trainer_thread = Process(target=agent.train_in_loop, daemon=True)
    trainer_thread.start()
    # while not agent.training_initialized:
    #     time.sleep(0.1)
    
    rewards = []
    episode_list = []
    
    try: 
        with open('_out/episode_list.pkl','rb') as f:
            episode_list = pickle.load(f)
        with open('_out/rewards_list.pkl','rb') as f:
            rewards = pickle.load(f)
    except:
        pass
        
    # Iterate over episodes
    for episode in tqdm(range(startEpisode, startEpisode+500), unit='episodes'):
        #try:
            pygame.init()
            env.collision_hist = []
            env.obstacle_data = []
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
            with CarlaSyncMode(env.world, env.rgb_sensor, env.sem_sensor, fps=20) as sync_mode:
                
                #add initial delay to speed up car spawn
                for i in range(50):
                    try:
                        snapshot, image_rgb, image_semantic = sync_mode.tick(timeout=20.0)
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
                    snapshot, image_rgb, image_semantic = sync_mode.tick(timeout=20.0) #This is resets tick
                except:
                    print("error")
                    break
                proc_start = time.time()
                semantic_segmentation = env.process_sem(image_semantic)
                proc_end = time.time()-proc_start

                current_state = semantic_segmentation

                while True:
                    #Visualizations#

                    if random.random() > epsilon:
                        # Get action from Q table
                        start = time.time()
                        action = torch.argmax(agent.get_qs(current_state)).item()
                        #print("model time: ", time.time()-start)
                    else:
                        # Get random action
                        action = random.randrange(3)
                        # This takes no time, so we add a delay matching 60 FPS (prediction above takes longer)
                        #time.sleep(0.04)

                    ## For one action, apply it twice so the car can actually apply angle 
                    for i in range(2):
                        new_state, reward, done, err = env.step(action, sync_mode)
                    ##

                    if err is True:
                        break

                    # Transform new continous state to new discrete state and count reward
                    episode_reward += reward

                    # Every step we update replay memory
                    if step > 30:
                        agent.update_replay_memory((current_state, action, reward, new_state, done))

                    current_state = new_state
                    step += 1
                    print("step: "+ str(step)+ " reward: "+str(reward)+" action: ", str(action))
                    env.num_timesteps = step

                    if done:
                        break

            episode_list.append(episode)
            rewards.append(episode_reward)
            # End of episode - destroy agents
            for actor in env.actor_list:
                actor.destroy()
            #agent.train()
            pygame.quit()

            # Decay epsilon
            if epsilon > MIN_EPSILON:
                epsilon *= EPSILON_DECAY
                epsilon = max(MIN_EPSILON, epsilon)
            
            plt.figure(1)
            plt.xlabel('Episodes')
            plt.ylabel('Rewards')
            plt.title(str('Training Obstacle: Ground Truth Single Model'))  
            plt.plot(episode_list, rewards)
            plt.savefig('_out/reward_graph.png')

            # plt.figure(2)
            # plt.xlabel('Timesteps')
            # plt.ylabel('Epsilon')
            # plt.title(str('Epsilon Decay'))  
            # plt.plot(timesteps, rewards)
            # plt.savefig('_out/reward_graph.png')
    
    try: 
        with open('_out/episode_list.pkl','wb') as f:
            pickle.dump(episode_list,f)
        with open('_out/rewards_list.pkl','wb') as f:
            pickle.dump(rewards,f)
        with open('_out/replay_memory.pkl','wb') as f:
            pickle.dump(agent.replay_memory,f)

        state = {
            'epsilon': epsilon,
            'episode': episode+1,
            'model_state_dict': agent.model.state_dict(),
            'optimizer': agent.optimizer.state_dict()
                }

        torch.save(state, 'models/saved_model.pt')
    except:
        print("Error with pickle files")
        pass
        
    print("Epsilon at 500 episodes: ", epsilon)   
    print("Total Runtime for 500 Episodes: ", time.time()-process_start)
    # Set termination flag for training thread and wait for it to finish
    agent.terminate = True
    torch.cuda.empty_cache()
    #trainer_thread.join()
    trainer_thread.terminate()
    sys.exit("Training finished")
    #agent.model.save('models/all_waypoints_model')