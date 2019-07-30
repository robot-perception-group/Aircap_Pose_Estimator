from __future__ import print_function, division
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

import numpy as np
import os, sys
import torch
from torchSMPL import torchSMPL, torch_vposer
import param as config
if config.data == 'raw':
    from camera_and_NN2 import processCamsNNs
elif config.data == 'processed':
    from camera_and_NN import processCamsNNs
from torchCam import torchCam
import itertools
from utils import geman_mcclure
import tf

smpl2deepcut_joint_map = torch.tensor([8,5,2,1,4,7,21,19,17,16,18,20,12,15])

class aircapFitter():

    def __init__(self,loadRes=False,device='cpu',loadData=True,
                    smpl_path = 'torchSMPL/basicModel_m_lbs_10_207_0_v1.0.0.pkl',
                    data_root = '',
                    nn_list = ['alphapose'],
                    camlist = None,
                    **kwargs):
        
        self.device = device
        self.dtype = torch.float32

        logger.info('loading torchSMPL')
        self.smpl_model = torchSMPL.create(smpl_path,device=self.device)
        logger.info('torchSMPL loaded')

        self.loadData(data_root,camlist,nn_list)
        # Results arrays
        self.resTrans = torch.zeros([zip(*self.tstamps2cam)[1].count(0),3],dtype=self.dtype)
        self.resPose = torch.zeros([zip(*self.tstamps2cam)[1].count(0),72],dtype=self.dtype)        # Tx24X9 rotation matrices for each joint
        self.resShape = torch.zeros([self.resPose.shape[0],10],dtype=self.dtype)                    # Tx10
        self.resZvposer = torch.zeros([self.resPose.shape[0],32],dtype=self.dtype)
        self.resSmplRootRot = torch.zeros([self.resPose.shape[0],3],dtype=self.dtype)
        self.resCams = np.zeros([self.resPose.shape[0],self.n_cams,2,4,4])      # TxCx2x4x4     3rd dimension for original and optimized output
        self.resTstamps = [['']*self.n_cams for n in range(self.resPose.shape[0])]                       # TxC list
        self.resSmoothPose = torch.zeros([zip(*self.tstamps2cam)[1].count(0),72],dtype=self.dtype)        # Tx72
        self.resSmoothTrans = torch.zeros([self.resPose.shape[0],3],dtype=self.dtype)        # Tx3
        self.resSmoothShape = torch.zeros([self.resPose.shape[0],10],dtype=self.dtype)                    # Tx10


    def loadData(self,data_root,camlist,nn_list):
        self.nn_list = nn_list
        self.camlist = camlist
        self.n_cams,self.n_NNs,self.camsdata,self.NNs,self.tstamps2cam = processCamsNNs(data_root,nn_list,camlist)

        # SMPL prior
        self.vposer = torch_vposer.vposer_decoder()
        self.vposer.load_from_numpy('torchSMPL/vposerDecoderWeights.npz')
        if self.device is 'cuda':
            self.vposer = self.vposer.cuda()

        self.reset_vars(retain_vals=False)


    def reset_vars(self,retain_vals=True,loadForSmoothen=False,**kwargs):
        if retain_vals:
            
            self.smplTrans = self.smplTrans.clone().detach().requires_grad_(True)
            self.smplRootRot = self.smplRootRot.clone().detach().requires_grad_(True)
            self.smplPose = self.smplPose.clone().detach().requires_grad_(True)
            self.Zvposer = self.Zvposer.clone().detach().requires_grad_(True)

            # self.smplShape = torch.zeros(1,10,dtype=self.dtype,device=self.device,requires_grad=True)
             # Fixed SMPL shape
            self.smplShape = torch.tensor([[0.54127824,  0.4374679 , -0.52830036, -0.49878247,  1.12260918,
                                                 0.01637326, -1.43266301, -1.10238274,  1.5591364 , -0.10529658]],
                                                 dtype=self.dtype,device=self.device)
        else:
            self.smplTrans = torch.zeros(1,3,dtype=self.dtype,device=self.device,requires_grad=True)
            self.smplRootRot = torch.zeros(1,3,dtype=self.dtype,device=self.device,requires_grad=True)
            self.Zvposer = torch.zeros(1,32,dtype=self.dtype,device=self.device,requires_grad=True)
            self.smplPose = torch.zeros(1,69,dtype=self.dtype,device=self.device,requires_grad=True)
            # self.smplShape = torch.zeros(1,10,dtype=self.dtype,device=self.device,requires_grad=True)
            # Nitin's SMPL shape
            self.smplShape = torch.tensor([[0.54127824,  0.4374679 , -0.52830036, -0.49878247,  1.12260918,
                                                0.01637326, -1.43266301, -1.10238274,  1.5591364 , -0.10529658]],
                                                dtype=self.dtype,device=self.device)

        # observation arrays
        self.Js = torch.zeros([self.n_cams,self.n_NNs,24,2],device=self.device,dtype=self.dtype)
        self.Ws = torch.zeros([self.n_cams,self.n_NNs,24],device=self.device,dtype=self.dtype)
        self.refTrans = torch.zeros([self.n_cams,3],device=self.device,dtype=self.dtype)
        self.refRot = torch.zeros([self.n_cams,3],device=self.device,dtype=self.dtype)

        self.torchCams = []
        for cam in range(self.n_cams):
            # all cams have same intrinsics
            self.torchCams.append(torchCam(self.camsdata[cam].get_intrinsic(),device=self.device))
        

    
    def LossFunctionvposer(self):
        smplPose = torch.cat([(self.smplRootRot),self.vposer.forward(self.Zvposer)],1)
        # get 3D joints
        j3d = self.smpl_model.forward(trans=self.smplTrans,pose=smplPose,betas=self.smplShape)
        
        # project j3d on camera
        loss2d = 0
        transLoss = 0
        rotLoss = 0
        for cam in range(self.n_cams):
            j2d = self.torchCams[cam].projectOn2d(j3d)
            for n in range(self.n_NNs):
                # loss2d += torch.sum(self.Ws[cam,n].unsqueeze(1)*((j2d-self.Js[cam,n])**2))
                loss2d += torch.sum(self.Ws[cam,n].unsqueeze(1)*(geman_mcclure(j2d-self.Js[cam,n],self.gemanSigma)))
            transLoss += torch.sum(geman_mcclure(self.torchCams[cam].translation - self.refTrans[cam],1))
            rotLoss += torch.sum(geman_mcclure(self.torchCams[cam].rotation - self.refRot[cam],1))

        betaRegLoss = torch.sum(self.smplShape**2)
        posePriorReg = torch.sum(self.Zvposer**2)    # Full pose is [root trans, root rot, ...]

        logger.debug('loss2d : %s ; betaloss : %s ; posePriorRegLoss : %s  ;  transLoss : %s ; rotLoss : %s',
                            loss2d.data.cpu().numpy(),
                            betaRegLoss.data.cpu().numpy(),
                            posePriorReg.data.cpu().numpy(),
                            transLoss.data.cpu().numpy(),
                            transLoss.data.cpu().numpy())

        lossFinal = self.loss2dWeight*loss2d + self.betalossWeight*betaRegLoss  \
            + self.priorlossWeight*posePriorReg + self.transLossWeight*transLoss + self.rotLossWeight*rotLoss

        self.optim.zero_grad()
        lossFinal.backward()

        return lossFinal

    def fit(self,idxWin=[0,None],
                lossW = [10, 0.1, 1,10,10],gemanSigma=40,
                optim_param=[0.25,0.15],optimiters=[2000,100],
                optimName='Adam',
                poseprior = 'vposer',
                retain_vals=False,
                camopt=True):

        # loss function weights
        self.idxWin = idxWin
        self.loss2dWeight = lossW[0]
        self.betalossWeight = lossW[1]
        self.priorlossWeight = lossW[2]
        self.transLossWeight = lossW[3]
        self.rotLossWeight = lossW[4]
        self.gemanSigma = gemanSigma
        self.optimlr = optim_param[0]
        self.optimiters = optimiters[0]
        self.optimName = optimName
        self.poseprior = poseprior
        self.camopt = camopt
        # self.camLossWeight = lossW[3]
        if self.poseprior == 'vposer':
            self.LossFunction = self.LossFunctionvposer
        else:
            print('Invalid poseprior')

        # reset variables
        self.reset_vars(retain_vals)

        # cam 0 timestamps
        cam0_tstamps = [t[0] for t in self.tstamps2cam if t[1] == 0]        # # t[0] is timestamp ; t[1] is corresponding camera
        cam0_tstamps = cam0_tstamps[self.idxWin[0]:self.idxWin[1]]


        self.extrinsic_offset = np.zeros([self.n_cams,3])
        # Estimation for each time stamp
        idx=0
        t = cam0_tstamps[0]
        self.fitSingle(idx,t)
        self.optimiters = optimiters[1]
        self.optimlr = optim_param[1]
        for idx,t in enumerate(cam0_tstamps[1:]):
            self.fitSingle(idx+1,t)

    def fitSingle(self,idx,t):

        
        self.reset_vars(retain_vals=True)

        timestamp = t
        presentCam = 0

        # import ipdb; ipdb.set_trace()

        extrinsic,sigmaExtrinsic = self.camsdata[presentCam].get_extrinsic_and_cov(timestamp)
        self.resCams[idx,presentCam,0,:,:] = extrinsic
        self.refTrans.data[presentCam] = torch.from_numpy(extrinsic[:3,3]).float().to(self.Js.device)
        self.refRot.data[presentCam] = torch.from_numpy(np.array(tf.transformations.euler_from_matrix(extrinsic))).float().to(self.Js.device)
        self.resTstamps[idx][0] = timestamp
        # set camera extrinsic and uncertainty
        extrinsic[:3,3] += self.extrinsic_offset[presentCam]
        self.torchCams[presentCam].setExtrinsic(extrinsic)
        self.torchCams[presentCam].setSigmaExtrinsic(sigmaExtrinsic)

        # load 2D joint observation
        for n in range(self.n_NNs):
            j,w,_ = self.NNs[n][presentCam].get_2d_joints_and_probs(timestamp,self.camsdata[presentCam].roi[timestamp])
            self.Js.data[presentCam,n] = torch.from_numpy(j.copy()).float().to(self.Js.device)
            self.Ws.data[presentCam,n] = torch.from_numpy(w.copy()).float().to(self.Ws.device)
            
        for cam in range(1,self.n_cams):
            timestamp = self.camsdata[cam].get_closest_time_stamp(t)
            presentCam = cam
            extrinsic,sigmaExtrinsic = self.camsdata[presentCam].get_extrinsic_and_cov(timestamp)
            self.resCams[idx,presentCam,0,:,:] = extrinsic
            self.refTrans.data[presentCam] = torch.from_numpy(extrinsic[:3,3]).float().to(self.Js.device)
            self.refRot.data[presentCam] = torch.from_numpy(np.array(tf.transformations.euler_from_matrix(extrinsic))).float().to(self.Js.device)
            self.resTstamps[idx][presentCam] = timestamp
            # set camera extrinsic and uncertainty
            extrinsic[:3,3] += self.extrinsic_offset[presentCam]
            self.torchCams[presentCam].setExtrinsic(extrinsic)
            self.torchCams[presentCam].setSigmaExtrinsic(sigmaExtrinsic)
            # load 2D joint observation
            for n in range(self.n_NNs):
                j,w,_ = self.NNs[n][presentCam].get_2d_joints_and_probs(timestamp,self.camsdata[presentCam].roi[timestamp])
                self.Js.data[presentCam,n] = torch.from_numpy(j.copy()).float().to(self.Js.device)
                self.Ws.data[presentCam,n] = torch.from_numpy(w.copy()).float().to(self.Ws.device)

        # Parameters to be optimized
        self.optimList = []
        if self.camopt:
            for cam in range(self.n_cams):
                self.optimList.append(self.torchCams[cam].rotation)
                self.optimList.append(self.torchCams[cam].translation)
        self.optimList.append(self.smplTrans)
        self.optimList.append(self.smplRootRot)
        #self.optimList.append(self.smplShape)
        if self.poseprior == 'vposer':
            self.optimList.append(self.Zvposer)
        elif self.poseprior == 'gmm':
            self.optimList.append(self.smplPose)

        # optimizer
        if self.optimName == 'SGD':
            self.optim = torch.optim.SGD(self.optimList,lr = self.optimlr)
        elif self.optimName == 'Adam':
            self.optim = torch.optim.Adam(self.optimList,lr = self.optimlr)
        elif self.optimName == 'LBFGS':
            self.optim = torch.optim.LBFGS(self.optimList,lr = self.optimlr)

        # optimization loop
        for n_iter in range(self.optimiters):
            loss = self.optim.step(self.LossFunction)
            logger.debug('AllOpt; Tstamp : %s ; iteration : %s ; loss : %s',idx,n_iter,loss.data.cpu().numpy())
        logger.info('AllOpt; Tstamp : %s ; loss : %s',idx,loss.data.cpu().numpy()) 


        with torch.no_grad():
            if self.poseprior == 'vposer':
                smplPose = torch.cat([(self.smplRootRot),self.vposer.forward(self.Zvposer)],1)

        # Save results
        self.resTrans[idx] = self.smplTrans.data[0]
        self.resPose[idx] = smplPose.data[0]
        self.resShape[idx] = self.smplShape.data[0]
        self.resZvposer[idx] = self.Zvposer.data[0]
        self.resSmplRootRot[idx] = self.smplRootRot.data[0]
        for cam in range(self.n_cams):
            self.resCams[idx,cam,1,:,:] = self.torchCams[cam].extrinsics().data.cpu().numpy()
            self.extrinsic_offset[cam] = (self.resCams[idx,cam,1,:,:] - self.resCams[idx,cam,0,:,:])[:3,3]
            

    def save_results(self,file_name):
        np.savez(file_name,
            trans=self.resTrans.data.cpu().numpy(),
            pose=self.resPose.data.cpu().numpy(),
            zvposer = self.resZvposer.data.cpu().numpy(),
            smplrootrot = self.resSmplRootRot.data.cpu().numpy(),
            shape=self.resShape.data.cpu().numpy(),
            cams=self.resCams,
            tstamps=self.resTstamps,
            idxWin = self.idxWin,
            smoothTrans=self.resSmoothTrans.data.cpu().numpy(),
            smoothPose=self.resSmoothPose.data.cpu().numpy(),
            smoothShape=self.resSmoothShape.data.cpu().numpy())

