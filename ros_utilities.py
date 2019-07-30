"""This module contains utilities around ROS
"""

from datetime import datetime
import xml.etree.ElementTree as et
import numpy as np


def get_time_of_file(mtime):
    """Returns the datetime instance associated to an mtime from our ROS setup

    the mtimes are usually encoded in the file names.
    File names are encoded in mtime with nanoseconds precision"""

    return datetime.fromtimestamp(mtime / 1000000000.0)


def get_extrinsic_param_from_file(path):
    '''
    Takes path of the xml file as input argument and returns the inverse of camera extrinsic parameters
    in world homogeneous coordinates
    '''

    import tf

    # parse the file and get the tree
    tree = et.parse(path)

    # get the root
    root = tree.getroot()
    
    # get the position of the camera (indices are hardcoded for this tree structure)
    pos = np.array([float(root[1][0][0][0].text),
                    float(root[1][0][0][1].text),
                    float(root[1][0][0][2].text)])

    # get the orientation of the camera in quaternion (indices are harcoded for this tree structure)
    orient_quat = np.array([float(root[1][0][1][0].text),
                            float(root[1][0][1][1].text),
                            float(root[1][0][1][2].text),
                            float(root[1][0][1][3].text)])

    # covariance matrix
    cov_mat = np.loadtxt(root[1][1].text[1:-1].split(','))
    cov_mat = cov_mat.reshape(6,6)

    # import ipdb;ipdb.set_trace()

    # convert quaternion to homogeneous rotation matrix (4X4)
    orient = tf.transformations.quaternion_matrix(orient_quat)

    ext = orient
    # merge the rotation and translation
    ext[0:3, -1] = pos

    # return the inverse of this camera extrinsic matrix
    return np.linalg.inv(ext), ext, cov_mat


def get_intrinsic_param_from_file(path):
    '''
    Takes the path of the xml file as argument and returns the camera intrinsic parameters
    '''
    # parse the file and get the tree
    tree = et.parse(path)

    # get the root
    root = tree.getroot()

    # get the intrinsic parameters (K from the ROS CameraInfo message) (indices are hardcoded for this tree structure)
    split_txt = root[5].text[1:-1].split(',')
    intrinsic = np.reshape(np.array([float(elem) for elem in split_txt]), [3, 3])

    return intrinsic
