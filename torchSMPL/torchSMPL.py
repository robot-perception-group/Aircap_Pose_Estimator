from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os

import pickle


import numpy as np
#  import tensorflow as tf

import torch
import torch.nn as nn

from smpl.smpl_webuser.serialization import load_model
from .torch_lbs import lbs

smpl2deepcut_joint_map = torch.tensor([8,5,2,1,4,7,21,19,17,16,18,20,12,15])

def create(model_path, model_type='smpl', **kwargs):
    if model_type.lower() == 'smpl':
        return SMPL(model_path, **kwargs)
    else:
        print('Unknown model type, exiting!')
        sys.exit(-1)


def to_np(array, dtype=np.float32):
    return np.array(array, dtype=dtype)


def to_tensor(array, dtype=torch.float32,device='cpu'):
    return torch.tensor(array, dtype=dtype,device=device)


class SMPL(nn.Module):

    NUM_JOINTS = 23

    def __init__(self, model_path, betas=None,
                 global_pose=None,
                 pose=None,
                 dtype=torch.float32,
                 num_betas=10,
                 use_face_keypoints=True,
                 use_hand_keypoints=False,
                 batch_size=1,
                 use_sparse=False,
                 joint_maps=None,
                 device=None,
                 **kwargs):
        '''
            Keyword Arguments:
                - use_hand_keypoints: Use vertices on the hands as extra joints
        '''
        super(SMPL, self).__init__()
        self.use_face_keypoints = use_face_keypoints
        self.use_hand_keypoints = use_hand_keypoints
        self.joint_maps = joint_maps

        #  with open(model_path, 'r') as model_file:
        model = load_model(model_path)
        self.faces = model.f

        if device is None:
            device = torch.device('cpu')

        # The shape coefficients
        if betas is None:
            betas = torch.zeros([batch_size, num_betas], dtype=dtype,
                                device=device)
        self.register_parameter('betas',
                                nn.Parameter(betas))

        # The tensor that contains the global pose of the model
        # It is separated from the pose of the joints in case we wish to
        # optimize only over one of them
        if global_pose is None:
            global_pose = torch.zeros([batch_size, 3], dtype=dtype,device=device)
        self.register_parameter('global_pose',
                                nn.Parameter(global_pose))
        # The tensor that contains the pose of the joints
        if pose is None:
            pose = torch.zeros([batch_size, (self.NUM_JOINTS) * 3],
                               dtype=dtype,device=device)
        self.register_parameter('pose',
                                nn.Parameter(pose))

        # The vertices of the template model
        self.register_buffer('v_template', to_tensor(to_np(model.v_template),device=device))
        # The shape components
        self.register_buffer('shapedirs',
                             to_tensor(to_np(model.shapedirs),device=device))

        j_regressor = model.J_regressor if use_sparse else \
            to_tensor(to_np(model.J_regressor.todense()),device=device)
        self.register_buffer('J_regressor', j_regressor)

        # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*3 x 207
        num_pose_basis = model.posedirs.shape[-1]
        # 207 x 20670
        posedirs = np.reshape(model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs', to_tensor(to_np(posedirs),device=device))

        # indices of parents for each joints
        parents = to_tensor(to_np(model.kintree_table[0]),device=device).long()
        self.register_buffer('parents', parents)

        self.register_buffer('lbs_weights', to_tensor(to_np(model.weights),device=device))

    def forward(self, trans, pose,in_global=True, 
                    for_deepcut=False,
                    get_skin=False,
                    betas = None, *args, **kwargs):
        
        if pose is None:
            pose = torch.cat([self.global_pose, self.pose], dim=1)
            
        if betas is None:
            betas = self.betas
        
        batch_size = pose.shape[0]

        vertices, joints = lbs(betas, pose, self.v_template,
                               self.shapedirs, self.posedirs,
                               self.J_regressor, self.parents,
                               self.lbs_weights,
                               batch_size=batch_size,**kwargs)

        if self.joint_maps is not None:
            joints = joints[:, self.joint_maps]

        if in_global:
            joints_global = joints + trans[:,:3].view(-1,1,3)
            if for_deepcut:
                return joints_global[:,smpl2deepcut_joint_map,:]
            if get_skin:
                vertices_global = vertices + trans[:,:3].view(-1,1,3)
                return vertices_global, joints_global
            else:
                return joints_global
            
        if get_skin:
            return vertices, joints
        else:
            return joints




def create_angle_prior(prior_type):
    if prior_type == 'smplify':
        return SMPLifyAnglePrior()
    else:
        err_msg = 'Unknown angle prior  type: {}'.format(prior_type)
        raise NotImplementedError(err_msg)


class SMPLifyAnglePrior(nn.Module):
    def __init__(self):
        super(SMPLifyAnglePrior, self).__init__()

        # Indices for the roration angle of
        # 55: left elbow,  90deg bend at -np.pi/2
        # 58: right elbow, 90deg bend at np.pi/2
        # 12: left knee,   90deg bend at np.pi/2
        # 15: right knee,  90deg bend at np.pi/2
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        angle_prior_idxs = np.array([55, 58, 12, 15], dtype=np.int64)
        angle_prior_idxs = torch.tensor(angle_prior_idxs, dtype=torch.long,device=device)
        self.register_buffer('angle_prior_idxs', angle_prior_idxs)

        angle_prior_signs = np.array([1, -1, -1, -1], dtype=np.float32)
        angle_prior_signs = torch.tensor(angle_prior_signs,
                                         dtype=torch.float32, device=device)
        self.register_buffer('angle_prior_signs', angle_prior_signs)

    def forward(self, pose, with_global_pose=False):
        ''' Returns the angle prior loss for the given pose

        Args:
            pose: (Bx[23 + 1] * 3) torch tensor with the axis-angle
            representation of the rotations of the joints of the SMPL model.
        Kwargs:
            with_global_pose: Whether the pose vector also contains the global
            orientation of the SMPL model. If not then the indices must be
            corrected.
        Returns:
            A sze (B) tensor containing the angle prior loss for each element
            in the batch.
        '''
        angle_prior_idxs = self.angle_prior_idxs - (not with_global_pose) * 3
        return torch.exp(pose[:, angle_prior_idxs] *
                         self.angle_prior_signs).sum(dim=1)
