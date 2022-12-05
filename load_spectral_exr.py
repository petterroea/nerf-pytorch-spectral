import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2
import OpenEXR
import Imath

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

def write_exr(fname, data, chnames, skip_chnames):
    print("Writing: %s" % str(data))

    processed_data = (data * float(2**16)).astype('uint16')

    w, h = data.shape[1], data.shape[0]
    print("Generating OpenEXR image with w %s h %s ch %s" % (w, h, data.shape[2]))

    header = OpenEXR.Header(w, h)

    channelDict = {}
    dataDict = {}
    for idx, channel in enumerate(chnames):
        channelDict[channel] = Imath.Channel(
            Imath.PixelType(Imath.PixelType.HALF),
            1,
            1
        )
        channel_data = processed_data[:,:,idx]
        channel_data = channel_data.flatten()
        dataDict[channel] = channel_data.tobytes()
        #print(dataDict[channel])
    
    if skip_chnames is not None:
        for idx, channel in enumerate(skip_chnames):
            print("Filling %s with zeroes" % channel)
            channelDict[channel] = Imath.Channel(
                Imath.PixelType(Imath.PixelType.HALF),
                1,
                1
            )
            fillerdata = np.zeros((h, w), dtype='uint16')
            dataDict[channel] = fillerdata.flatten().tobytes()
    
    header['channels'] = channelDict

    out = OpenEXR.OutputFile(fname, header)

    out.writePixels(dataDict)
    out.close()

# Assumes all channels are in the format HALF()
def load_exr(fname, chskip=3):
    print(fname)
    file = OpenEXR.InputFile(fname)
    header = file.header()

    chnames = list(header['channels'].keys())
    # Avoid dark channels?
    skipchans = []
    if chskip != 0:
        skipchans = chnames[-chskip:]
        chnames = chnames[:-chskip]

    data_window = header['dataWindow']
    #print(chnames)
    channelcount = len(chnames)

    w = data_window.max.x + 1
    h = data_window.max.y + 1
    print("Loading OpenEXR image with w %s h %s ch %s(skip %s)" % (w, h, channelcount, chskip))
    #print("Image size: %s x %s, channels: %s" % (w, h, channelcount))
    channels = file.channels(chnames)
    channels_np = np.array([ np.frombuffer(channel, dtype="uint16").astype("float32")/float(2**16) for channel in channels ])

    # TODO is this the correct major axis?
    data = np.empty((h, w, channelcount), dtype='float32')
    for rowno in range(h):
        for col in range(w):
            data[rowno, col, :] = channels_np[:, rowno*w+col]
    
    return chnames, skipchans, data


def load_exr_data(basedir, half_res=False, testskip=1, exr_chskip=3):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]

    exr_channels = None
    exr_skip_channels = None
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
            
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.exr')

            channels, skipchans, imgdata = load_exr(fname, exr_chskip)
            imgs.append(imgdata)

            if exr_channels == None:
                exr_channels = channels
                exr_skip_channels = skipchans
                print("Set channels(%d): %s" % (len(channels), channels))
            else:
                if channels != exr_channels:
                    raise RuntimeError("Got EXR image with channel list %s - expected %s" % (exr_channels, channels))

            poses.append(np.array(frame['transform_matrix']))
        #imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        imgs = np.array(imgs)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

        
    return exr_channels, exr_skip_channels, imgs, poses, render_poses, [H, W, focal], i_split


