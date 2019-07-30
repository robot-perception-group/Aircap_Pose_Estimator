from __future__ import division
import numpy as np
import os
import cv2


def _npcircle(image, cx, cy, radius, color, transparency=0.0):
    """Draw a circle on an image using only numpy methods."""
    radius = int(radius)
    cx = int(cx)
    cy = int(cy)
    y, x = np.ogrid[-radius: radius, -radius: radius]
    index = x ** 2 + y ** 2 <= radius ** 2
    # import pdb;pdb.set_trace()
    image[cy - radius:cy + radius, cx - radius:cx + radius][index] = (
        image[cy - radius:cy + radius, cx - radius:cx + radius][index].astype('float32') * transparency +
        np.array(color).astype('float32') * (1.0 - transparency)).astype('uint8')


def visJoints(joints,image,imageInputType='path',jointsInputType='path'):
    if imageInputType == 'path':
        img = cv2.imread(image, 1)
    else:
        img = image.copy()

    if jointsInputType == 'path':
            points = np.load(joints)
    else:
        points = joints


    colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 245, 255], [255, 131, 250], [255, 255, 0],
                  [255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 245, 255], [255, 131, 250], [255, 255, 0],
                      [0, 0, 0], [255, 255, 255]]
        
    for idx in np.arange(len(colors)):
        if ((points[1, idx] < img.shape[0] - 16) and (points[0, idx] < img.shape[1] - 16)
                    and (points[0, idx] > 16) and (points[1, idx] > 16)):
            _npcircle(img, points[0, idx], points[1, idx], 8, colors[idx], 0.0)

    return img


def geman_mcclure(error,sigma):
    return (error**2)/((sigma**2)+(error**2))