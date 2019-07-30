from torchSMPL import smpl_anim
import sys, os
import numpy as np

res_dir = sys.argv[1]

res = np.load(os.path.join(res_dir,'results.npz'))
err = np.load(os.path.join(res_dir,'err_res.npz'))

pose_aircap = res['pose']

dxsens = np.load('data/xsens_tstamped.npz')
xpose = dxsens['syncpose'][err['seq_start']:8000]

pose_aircap = pose_aircap[:xpose.shape[0]]
pose_aircap[:,:3] = 0
xpose[:,:6] = 0
xpose = xpose[:,3:]

xtrans = np.zeros([xpose.shape[0],3])
atrans = np.zeros([xpose.shape[0],3])
atrans[:,0] = 1

smpl_anim.smpl_anim2(pose=[xpose,pose_aircap],trans=[xtrans,atrans])

