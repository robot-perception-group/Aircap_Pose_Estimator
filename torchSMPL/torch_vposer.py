from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as torchdist
import torchgeometry as tgm


class vposer_encoder(nn.Module):

    def __init__(self, latentD=32, hidden_dim=512, is_training=False):
        super(vposer_encoder, self).__init__()

        self.is_training = is_training
        self.fc1 = nn.Linear(23*9, 512)
        self.fc2 = nn.Linear(512, 512)
        self.Zmu = nn.Linear(512, latentD)
        self.Zsigma = nn.Linear(512, latentD)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        fc1_out = F.leaky_relu(self.fc1(x))
        fc1_out = F.dropout(fc1_out, p=0.2, training=self.is_training)
        fc2_out = F.leaky_relu(self.fc2(fc1_out))
        Zmu_op = self.Zmu(fc2_out)
        Zsigma_op = F.softplus(self.Zsigma(fc2_out))

        return torchdist.Normal(loc=Zmu_op, scale=Zsigma_op)


class vposer_decoder(nn.Module):

    def __init__(self, latentD=32, is_training=False, dtype=torch.float32):
        super(vposer_decoder, self).__init__()
        self.dtype = dtype
        self.is_training = is_training
        self.fc1 = nn.Linear(latentD, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 23*9)

        #  self.projector = RodriguesProjector()

    def forward(self, x):
        fc1_out = F.leaky_relu(self.fc1(x), 0.2)
        fc1_out = F.dropout(fc1_out, p=0.25, training=self.is_training)
        fc2_out = F.leaky_relu(self.fc2(fc1_out), 0.2)
        fc3_out = torch.tanh(self.fc3(fc2_out))
        #  output = torch.reshape(fc3_out, shape=(-1, 23, 3, 3))[:, :-2, :]
        output = torch.reshape(fc3_out, shape=(-1, 3, 3))
        batch_size = x.shape[0]

        # Before converting the output rotation matrices of the VAE to
        # axis-angle representation, we first need to make them in to valid
        # rotation matrices
        with torch.no_grad():
            # Iterate over the batch dimension and compute the SVD
            norm_rotation = torch.zeros_like(output)
            for bidx in range(output.shape[0]):
                U, _, V = torch.svd(output[bidx])
                # Multiply the U, V matrices to get the closest orthonormal
                # matrix
                norm_rotation[bidx] = torch.matmul(U, V.t())

        # torch.svd supports backprop only for full-rank matrices.
        # The output is calculated as the valid rotation matrix plus the
        # output minus the detached output. If one writes down the
        # computational graph for this operation, it will become clear the the
        # output is the desired valid rotation matrix, while for the backward
        # pass gradients are propagated only to the original matrix
        # Source: PyTorch Gumbel-Softmax hard sampling
        correct_rot = norm_rotation - output.detach() + output

        return tgm.rotation_matrix_to_angle_axis(
            F.pad(correct_rot.view(-1, 3, 3), [0, 1, 0, 0])).view(batch_size, -1)

    def load_from_numpy(self, file_path):
        import numpy as np
        data = np.load(file_path)
        # TF data is 32X512 while pytorch layer data has format 512X32
        # that is why transpose
        self.fc1.weight.data = torch.from_numpy(data['fc1W'].T).to(self.dtype)
        self.fc1.bias.data = torch.from_numpy(data['fc1b']).to(self.dtype)
        self.fc2.weight.data = torch.from_numpy(data['fc2W'].T).to(self.dtype)
        self.fc2.bias.data = torch.from_numpy(data['fc2b']).to(self.dtype)
        self.fc3.weight.data = torch.from_numpy(data['fc3W'].T).to(self.dtype)
        self.fc3.bias.data = torch.from_numpy(data['fc3b']).to(self.dtype)
