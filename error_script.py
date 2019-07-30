import numpy as np
import sys
from torchSMPL import torchSMPL
import torch
import os

joint_map = [0,1,4,7,10,2,5,8,11,3,6,9,12,15,13,16,18,20,22,14,17,19,21,23]

def err_func(aircap_file,data,aircap_dir):
    if data == 'raw':
        seq_start = 28
    elif data == 'processed':
        seq_start = 0

    smpl_path = 'torchSMPL/basicModel_m_lbs_10_207_0_v1.0.0.pkl'
    smpl_model = torchSMPL.create(smpl_path)

    dxsens = np.load('data/xsens_tstamped.npz')
    xpose = dxsens['syncpose'][seq_start:8000]
    txpose = torch.tensor(xpose,dtype=torch.float32)
    # xtrans = dxsens['syncpose'][:,:3]

    aircap_res = np.load(aircap_file)
    apose = np.concatenate([aircap_res['trans'][:xpose.shape[0]],aircap_res['pose'][:xpose.shape[0]]],1)
    tapose = torch.tensor(apose,dtype=torch.float32)
    tashape = torch.tensor(aircap_res['shape'],dtype=torch.float32)

    # zero trans and root rot
    tempapose = tapose[:,3:].detach().clone()
    tempapose[:,:3] = 0
    tempatrans = tapose[:,:3].detach().clone()
    tempatrans[:,:] = 0
    averts_z,ajoints_z = smpl_model.forward(trans=tempatrans,pose=tempapose,betas=tashape[0:1],get_skin=True)
    print('aircap zero done')
    tempxpose = txpose[:,3:].detach().clone()
    tempxpose[:,:3] = 0
    tempxtrans = txpose[:,:3].detach().clone()
    tempxtrans[:,:] = 0
    xverts_z,xjoints_z = smpl_model.forward(trans=tempxtrans,pose=tempxpose,betas=tashape[0:1],get_skin=True)
    print('xsens zero done')


    # # errors
    errjoint_z = np.mean(abs(xjoints_z.data.numpy()-ajoints_z.data.numpy()),0)
    errverts_z = np.mean(abs(xverts_z.data.numpy()-averts_z.data.numpy()),0)

    errjoint_zt = np.mean(abs(xjoints_z.data.numpy()-ajoints_z.data.numpy())[:,1:],1)
    errverts_zt = np.mean(abs(xverts_z.data.numpy()-averts_z.data.numpy()),1)

    xjoints_z=xjoints_z.data.numpy()
    ajoints_z=ajoints_z.data.numpy()

    err = np.mean(np.linalg.norm((xjoints_z-ajoints_z),axis=2),0)

    np.save(aircap_dir+'/final_err_res',err[joint_map])

    np.savez(aircap_dir+'/err_res',xsens_joints=xjoints_z,
                                    aircap_joints=ajoints_z,
                                    aircap_angles=tempapose.data.numpy(),
                                    xsens_angles=tempxpose.data.numpy())

    print(aircap_dir)