import torch
import torch.nn as nn
import numpy as np



class torchCam(nn.Module):

    def __init__(self,intrinsic,orientation=None,translation=None,dtype=torch.float32,device=None):
        '''
        :param intrinsic: intrinsic parameters of the camera (3X3)
        :param orientation: rotation parameters (yaw, pitch, roll) of the camera (3)
        '''
        super(torchCam,self).__init__()

        if device is None:
            device = 'cpu'

        self.intrinsic = torch.tensor(intrinsic,dtype=dtype,device=device)
        if orientation is None:
            self.rotation = torch.zeros(3,dtype=dtype,device=device,requires_grad=True)
        else:
            self.rotation = torch.tensor(orientation,dtype=dtype,device=device,requires_grad=True)

        if translation is None:
            self.translation = torch.zeros(3,dtype=dtype,device=device,requires_grad=True)
        else:
            self.translation = torch.tensor(translation,dtype=dtype,device=device,requires_grad=True)


    def projectOn2d(self,points3d):
        '''
        Takes 3d points and intrinsic and extrinsic parameters
        of a camera. Gives the projected 2D point on the image plane.
        :param points: 3D points (NX3)
        '''
        
        extr_intr_mul = torch.matmul(self.intrinsic,torch.cat((self.rotationMatrix(),self.translation.view(3,-1)),dim=1))
        
        points2d = torch.matmul(extr_intr_mul[:3,:3],points3d.unsqueeze(-1)).squeeze(-1) + extr_intr_mul[:,3]
        
        return points2d[:,:,:2]/points2d[:,:,2:3]


    def setExtrinsic(self,extrinsic):
        
        self.rotation.data = torch.from_numpy(self.mat2eul(extrinsic[:3,:3])).float().to(self.rotation.device)                              
        self.translation.data = torch.from_numpy(extrinsic[:3,3]).float().to(self.translation.device)

    def setSigmaExtrinsic(self,sigma):
        self.InvSigmaExt = torch.from_numpy(np.linalg.inv(sigma.copy())).float().to(self.rotation.device)    
    
    def rotationMatrix(self):
        # self.rotation should be a 3X1 vector with euler angles in order of yaw, pitch, roll
        rot_matx = torch.tensor([[1,0,0],
                                [0,self.rotation[0].cos(),-self.rotation[0].sin()],
                                [0,self.rotation[0].sin(),self.rotation[0].cos()]],
                                dtype=self.rotation.dtype,device=self.rotation.device)

        rot_maty = torch.tensor([[self.rotation[1].cos(),0,self.rotation[1].sin()],
                                [0,1,0],
                                [-self.rotation[1].sin(),0,self.rotation[1].cos()]],
                                dtype=self.rotation.dtype,device=self.rotation.device)

        rot_matz = torch.tensor([[self.rotation[2].cos(),-self.rotation[2].sin(),0],
                                [self.rotation[2].sin(),self.rotation[2].cos(),0],
                                [0,0,1]],
                                dtype=self.rotation.dtype,device=self.rotation.device)

        return torch.matmul(rot_matz,torch.matmul(rot_maty,rot_matx))

    def extrinsics(self):
        return torch.cat((torch.cat((self.rotationMatrix(),
                            self.translation.unsqueeze(1)),1),
                            torch.tensor([[0,0,0,1]],device=self.translation.device,
                            dtype=self.translation.dtype)))


    def mat2eul(self,rot_mat):
        # This function is used to initialize the euler angles. 
        # It uses tf module from ros and is not differentiable.

        import tf
        import numpy as np

        return np.array(tf.transformations.euler_from_matrix(rot_mat))