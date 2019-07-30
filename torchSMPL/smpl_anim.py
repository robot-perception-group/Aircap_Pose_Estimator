from mayavi import mlab
import torchSMPL
import torch

@mlab.animate(delay=10)
def anim(verts,s):
    for i in range(verts.shape[0]):
        s.mlab_source.x = verts[i,:,0].data.numpy()
        s.mlab_source.y = verts[i,:,1].data.numpy()
        s.mlab_source.z = verts[i,:,2].data.numpy()
        yield

def smpl_anim(pose,trans,shape=None):
    pose = torch.tensor(pose,dtype=torch.float32)
    trans = torch.tensor(trans,dtype=torch.float32)
    if shape is not None:
        shape = torch.tensor(shape,dtype=torch.float32)
        if shape.ndimension() == 1:
            shape = shape[None,:]
    smpl = torchSMPL.create('torchSMPL/basicModel_m_lbs_10_207_0_v1.0.0.pkl')
    msh,jnts = smpl.forward(pose=pose,trans=trans,betas=shape,get_skin=True)
    s = mlab.triangular_mesh(msh[0,:,0].data.numpy(),msh[0,:,1].data.numpy(),msh[0,:,2].data.numpy(),smpl.faces)
    anim(msh,s)
    mlab.show()


@mlab.animate(delay=10)
def anim2(verts,s):
    for i in range(verts[0].shape[0]):
        for j in range(len(s)):
            s[j].mlab_source.x = verts[j][i,:,0].data.numpy()
            s[j].mlab_source.y = verts[j][i,:,1].data.numpy()
            s[j].mlab_source.z = verts[j][i,:,2].data.numpy()
        yield

def smpl_anim2(pose,trans,shape=None):
    smpl = torchSMPL.create('torchSMPL/basicModel_m_lbs_10_207_0_v1.0.0.pkl')
	
    msh = []
    s = []
    for i in range(len(pose)):
        pose_tens = torch.tensor(pose[i],dtype=torch.float32)
        trans_tens = torch.tensor(trans[i],dtype=torch.float32)
        if shape is not None:
            shape_tens = torch.tensor(shape[i],dtype=torch.float32)
            if shape_tens.ndimension() == 1:
                shape_tens = shape_tens[None,:]
        else:
            shape_tens = None
        m,_ = smpl.forward(pose=pose_tens,trans=trans_tens,betas=shape_tens,get_skin=True)
        msh.append(m)
    	s.append(mlab.triangular_mesh(msh[i][0,:,0].data.numpy(),msh[i][0,:,1].data.numpy(),msh[i][0,:,2].data.numpy(),smpl.faces))

    anim2(msh,s)
    mlab.show()
