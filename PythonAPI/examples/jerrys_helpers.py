import numpy as np
import torch

Unlabelled = [0, 0, 0]   
Building = [70, 70, 70]
Fence = [100, 40, 40]
Other = [55, 90, 80]
Pedestrian = [220,20,60]
Pole = [153, 153, 153]
RoadLine = [157, 234, 50]
Road = [128, 64, 128]
SideWalk = [244, 35, 232]
Vegetation = [107, 142, 35]
Vehicles = [0, 0, 142]
Wall = [102, 102, 156]
TrafficSign = [220, 220, 0]
Sky = [70, 130, 180]
Ground = [81, 0, 81]
Bridge = [150, 100, 100]
RailTrack = [230, 150, 140]
GuardRail = [180, 165, 180]
TrafficLight = [250, 170, 30]
Static = [110, 190, 160]
Dynamic = [170, 120, 50]
Water = [45, 60, 150]
Terrain = [145, 170, 100]

label_colours = np.array([
    Unlabelled, Building, Fence, Other, Pedestrian, Pole, RoadLine, Road, SideWalk, Vegetation, Vehicles, Wall,
    TrafficSign, Sky, Ground, Bridge, RailTrack, GuardRail, TrafficLight, Static, Dynamic,
    Water, Terrain
])

def color_semantic(tensor):
    temp = tensor.numpy()
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(len(label_colours)):
        r[temp==l]=label_colours[l,0]
        g[temp==l]=label_colours[l,1]
        b[temp==l]=label_colours[l,2]
    semantic_rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    semantic_rgb[:,:,0] = (b/255.0)#[:,:,0]
    semantic_rgb[:,:,1] = (g/255.0)#[:,:,1]
    semantic_rgb[:,:,2] = (r/255.0)#[:,:,2]

    return semantic_rgb

def load_weights(model, fpath):
    print("loading weights '{}'".format(fpath))
    weights = torch.load(fpath)
    startEpoch = weights['startEpoch']
    model.load_state_dict(weights['state_dict'])
    print("loaded weights (lastEpoch {}, loss {}, error {})".format(startEpoch, weights['loss'], weights['error']))
    return startEpoch

def get_predictions(output_batch):
    bs,c,h,w = output_batch.size()
    tensor = output_batch.data
    values, indices = tensor.cpu().max(1)
    indices = indices.view(bs,h,w)
    return indices

    ### Uncertainty ####

def calc_entropy(pred):
    '''
    Calculates entropy. 
    '''
    return -np.sum(pred * np.log(pred), axis=1)

def calc_aleatoric(probs):
    '''
    Calculates aleatoric uncertainty of an image
    '''
    aleatoric = []
    for prob in probs:
        ent = [calc_entropy(p) for p in prob]
        aleatoric.append(np.mean(ent, axis=0))
    return aleatoric

def get_pixels(output):
    '''
    Reshapes the (num_classes, height, width) NN output into (height * width, num_classes) such that
    output[:, 0, 0] = reshaped_output[0]
    output[:, 0, 1] = reshaped_output[1]
    ...
    output[:, 359, 478] = reshaped_output[172798]
    output[:, 359, 479] = reshaped_output[172799]
    '''
    pixels = np.reshape(np.ravel(output), 
                        (output.shape[1] * output.shape[2], output.shape[0]), 
                        order="F")
    return pixels

def softmax(pred):
    '''
    Softmax function. Expects input of shape (172800, 12). Output has shape (172800, 12)
    '''
    e_x = np.exp(pred - np.amax(pred, axis=1, keepdims=True))
    prob = e_x / np.sum(e_x, axis=1, keepdims=True)
    return prob