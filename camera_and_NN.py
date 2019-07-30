"""This module contains the main Camera object for manipulating the information extracted from each drone
"""

import os

from ros_utilities import get_time_of_file
import numpy as np
import yaml
import json
import cPickle as cpk
import tf

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

class Camera(object):
    """Encodes the timed view from a camera"""

    # @staticmethod
    # def load_from_folder(folder_data):
    #     intrinsic_param_path_m1 = os.path.join(folder_data, 'caminfo')
    #     extrinsic_param_path_m1 = os.path.join(folder_data, 'campose')
    #     img_path_m1 = os.path.join(folder_data, 'cropped_img')  # this ideally should be the original images path
    #     roi_path_m1 = os.path.join(folder_data, 'roi')

    #     images = [f for f in os.listdir(img_path_m1) if os.path.isfile(os.path.join(img_path_m1, f))]
    #     mtimes = [os.path.splitext(os.path.basename(f))[0] for f in images]
    #     mtimes.sort()
    #     timestamps = [get_time_of_file(int(f)) for f in mtimes]

    #     intrinsic_files = dict((t, os.path.join(intrinsic_param_path_m1, m + '.txt')) for t, m in zip(timestamps, mtimes))
    #     extrinsic_files = dict((t, os.path.join(extrinsic_param_path_m1, m + '.txt')) for t, m in zip(timestamps, mtimes))
    #     image_files = dict((t, os.path.join(img_path_m1, m + '.png')) for t, m in zip(timestamps, mtimes))
    #     roi_files = dict((t, os.path.join(roi_path_m1, m + '.yml')) for t, m in zip(timestamps, mtimes))

    #     uav = Camera(basedir=folder_data,
    #                  timestamps=timestamps,
    #                  intrinsics=intrinsic_files,
    #                  extrinsics=extrinsic_files,
    #                  images=image_files,
    #                  roi=roi_files)
    #     return uav

    @staticmethod
    def load_from_folder(folder_data):
        img_path_m1 = os.path.join(folder_data, 'full_img')  # this ideally should be the original images path

        images = [f for f in os.listdir(img_path_m1) if os.path.isfile(os.path.join(img_path_m1, f))]
        mtimes = [os.path.splitext(os.path.basename(f))[0] for f in images]
        timestamps = [f for f in mtimes]
        image_files = dict((t, os.path.join(img_path_m1, m + '.png')) for t, m in zip(timestamps, mtimes))
        
        timestamps = np.load(os.path.join(folder_data,'timestamps.npy'))
        fl = open(os.path.join(folder_data,'roi.pkl'),'r')
        roi = cpk.load(fl)
        fl.close()
        fl = open(os.path.join(folder_data,'campose_raw.pkl'),'r')
        # fl = open(os.path.join(folder_data,'campose.pkl'),'r')              # for online one, temporary
        extrinsics = cpk.load(fl)
        fl.close()
        
        
        fl = open(os.path.join(folder_data,'caminfo.pkl'),'r')
        intrinsics = cpk.load(fl)
        fl.close()

        uav = Camera(basedir=folder_data,
                     timestamps=timestamps,
                     intrinsics=intrinsics,
                     extrinsics=extrinsics,
                     images=image_files,
                     roi=roi)
        return uav

    def __init__(self,
                 basedir,
                 timestamps,
                 intrinsics,
                 extrinsics,
                 images,
                 roi):
        self.basedir = basedir
        self.timestamps = sorted(timestamps)
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.images = images
        self.roi = roi
        # self._max_timestamp = max(self.timestamps)

    def get_closest_time_stamp(self, query_timestamp):
        """Returns the closest timestamp contained in this camera wrt. a query timestamp.

        .. note::

           The query is linear in complexity to the number of timestamps and is not efficient if
           efficiency is needed.
        """
        return min(self.timestamps, key=lambda x: abs(int(x) - int(query_timestamp)))

    def get_intrinsic(self, timestamp=None):
        """Returns the intrincs of the camera at a specific time stamp

         :param    timestamp: the query timestamp
         :returns: the intrinsic matrix at the specified timestamp
        """

        return np.array(self.intrinsics['camera_matrix']['data']).reshape([3,3])

    def get_extrinsic_and_cov(self, timestamp):
        """Returns the extrinsic of the camera at a giveen time stamp.

        .. note:: the timestamp should be part of the timestamps of this camera, otherwise
          a `KeyError` exception will be raised.

        :param timestamp: the query timestamp as a datetime object
        :returns: the extrincics matrix
        """
        # import ipdb; ipdb.set_trace()
        po = self.extrinsics[timestamp]['position']
        pos = np.array([po.x,po.y,po.z])
        ori = self.extrinsics[timestamp]['orientation']
        orient_quat = np.array([ori.x,ori.y,ori.z,ori.w])
        cov_mat = np.array(self.extrinsics[timestamp]['covariance']).reshape(6,6)

        # convert quaternion to homogeneous rotation matrix (4X4)
        orient = tf.transformations.quaternion_matrix(orient_quat)

        ext = orient
        # merge the rotation and translation
        ext[0:3, -1] = pos

        # return the inverse of this camera extrinsic matrix
        return np.linalg.inv(ext), cov_mat
        # return self.extrinsics[timestamp][0], self.extrinsics[timestamp][0]         # for online one temporary

    def get_frame(self, timestamp):
        """Returns the frame viewed by the camera at a given timestamp
        """
        import cv2
        return cv2.imread(self.images[timestamp])

def get_2D_points_using_ROI(roi, points, mode):
    '''
    This function calculates the 2D location of the joints in the complete image. It accepts the path of the yaml file containing ROI parameters and the joints location in ROI.

    :param points: 2D coordinates of joints in ROI (2XN)
    '''

    x = roi['x']  # x coordinate of ROI in the full image
    y = roi['y']  # y coordinate of ROI in the full image

    if mode == 'full':
        # points coordinates in full image
        points[0, :] += x
        points[1, :] += y
    elif mode == 'cropped':
        # points coordinates in cropped image
        points[0, :] -= x
        points[1, :] -= y
    else:
        raise Exception('wrong mode provided')

    return points


class NN(object):

    # @staticmethod
    # def load_from_folder(folder_data,nn):

    #     name = nn
        
    #     vis_path = os.path.join(folder_data, 'vis')
    #     out_path = os.path.join(folder_data, 'cropped_img')

    #     out = [f for f in os.listdir(out_path) if os.path.isfile(os.path.join(out_path, f))]
    #     mtimes = [f.split('.')[0] for f in out]
    #     timestamps = [get_time_of_file(int(f)) for f in mtimes]

    #     if name == 'openpose':
    #         out_files = dict((t, os.path.join(out_path, m + '.json')) for t, m in zip(timestamps, mtimes))    
    #     else:
    #         out_files = dict((t, os.path.join(out_path, m + '.png.npz')) for t, m in zip(timestamps, mtimes))
        
    #     vis_files = dict((t, os.path.join(vis_path, m + '.png')) for t, m in zip(timestamps, mtimes))
    #     tstamps_raw = dict((t, int(m)) for t, m in zip(timestamps, mtimes))

    #     network = NN(name = name,
    #                  basedir=folder_data,
    #                  timestamps=timestamps,
    #                  tstamps_raw = tstamps_raw,
    #                  out=out_files,
    #                  vis=vis_files)
    #     return network

    # def __init__(self,
    #              name,
    #              basedir,
    #              timestamps,
    #              tstamps_raw,
    #              out,
    #              vis):
    #     self.name = name
    #     self.basedir = basedir
    #     self.timestamps = set(timestamps)
    #     self.tstamps_raw = tstamps_raw
    #     self.out = out
    #     self.vis = vis

    @staticmethod
    def load_from_folder(folder_data,nn):

        name = nn
        
        vis_path = os.path.join(folder_data, 'vis')
        out_path = os.path.join(folder_data, 'cropped_img')

        out = [f for f in os.listdir(out_path) if os.path.isfile(os.path.join(out_path, f))]
        mtimes = [f.split('.')[0] for f in out]
        timestamps = [f for f in mtimes]

        if name == 'openpose' or name == 'alphapose':
            out_files = dict((t, os.path.join(out_path, m + '.json')) for t, m in zip(timestamps, mtimes))    
        else:
            out_files = dict((t, os.path.join(out_path, m + '.png.npz')) for t, m in zip(timestamps, mtimes))
        
        vis_files = dict((t, os.path.join(vis_path, m + '.png')) for t, m in zip(timestamps, mtimes))

        network = NN(name = name,
                     basedir=folder_data,
                     timestamps=timestamps,
                     out=out_files,
                     vis=vis_files)
        return network

    def __init__(self,
                 name,
                 basedir,
                 timestamps,
                 out,
                 vis):
        self.name = name
        self.basedir = basedir
        self.timestamps = sorted(timestamps)
        self.out = out
        self.vis = vis

        # using nose as head
        if self.name == 'openpose':
            self.map2smpl = np.array([8,12,9,-1,13,10,-1,14,11,-1,19,22,1,-1,-1,0,5,2,6,3,7,4,-1,-1])
        # using nose as head
        if self.name == 'alphapose':
            self.map2smpl = np.array([-1,11,8,-1,12,9,-1,13,10,-1,-1,-1,1,-1,-1,0,5,2,6,3,7,4,-1,-1])

    def get_2d_joints_and_probs(self, timestamp, roi):
        '''
        If 2d points are required in cropped image, pass roi as None
        '''
        if self.name == 'deepcut':
            k = np.load(self.out[timestamp])['pose']
            if roi == None:
                j2d = np.transpose(k[0:2,:])
            else:
                j2d = np.transpose(get_2D_points_using_ROI(roi,k[0:2,:],'full'))
            prob = k[2,:]
        elif self.name == 'hmr':
            k = np.load(self.out[timestamp])['pose']
            if roi == None:
                j2d = np.transpose(k[1::-1, :14])
            else:
                j2d = np.transpose(get_2D_points_using_ROI(roi, k[1::-1, :14],'full'))
            prob = 0.3*np.ones(14)
        elif self.name == 'openpose':
            try:
                f = open(self.out[timestamp])
                people = json.load(f)['people']
                num_people = len(people)
            except:
                num_people = 0

            if num_people > 0:
                k = np.array(people[0]['pose_keypoints_2d']).reshape([25,3]).T
                if roi == None:
                    j2d = np.transpose(k[0:2,self.map2smpl])
                else:
                    j2d = np.transpose(get_2D_points_using_ROI(roi,k[0:2,self.map2smpl],'full'))
                prob = k[2,self.map2smpl]

                # points we are not using; make them and the prob as 0
                j2d[self.map2smpl==-1,:] = 0
                prob[self.map2smpl==-1] = 0
            else:
                j2d = np.zeros([24,2])
                prob = np.zeros([24])
        elif self.name == 'alphapose':
            try:
                f = open(self.out[timestamp])
                people = json.load(f)['people']
                num_people = len(people)
            except:
                num_people = 0
                
            if num_people > 0:
                k = np.array(people[0]['pose_keypoints_2d']).reshape([18,3]).T
                if roi == None:
                    j2d = np.transpose(k[0:2,self.map2smpl])
                else:
                    j2d = np.transpose(get_2D_points_using_ROI(roi,k[0:2,self.map2smpl],'full'))
                prob = k[2,self.map2smpl]
                
                # points we are not using; make them and the prob as 0
                j2d[self.map2smpl==-1,:] = 0
                prob[self.map2smpl==-1] = 0
            else:
                j2d = np.zeros([24,2])
                prob = np.zeros([24])
            

        return j2d, prob, num_people

    def get_cov_j2d(self):
        '''
        '''
        return np.stack([np.eye(2)]*14)

    def get_viz(self,timestamp):
        '''
        '''
        import cv2
        return cv2.imread(self.vis[timestamp])



def process_cameras(data_root,camlist=None):

    if camlist is None:
        n_cams = len(os.listdir(os.path.join(data_root,'data')))
        camlist = range(n_cams)
    cams = []
    for cam in camlist:
        
        logger.debug('processing camera %s',cam)

        cams.append(Camera.load_from_folder(os.path.join(data_root,'data',('machine_'+str(cam+1)))))

    return cams

def process_NN(data_root,nn_name,camlist=None):

    if camlist is None:
        n_cams = len(os.listdir(os.path.join(data_root,'data')))
        camlist = range(n_cams)

    nn_root = os.path.join(data_root,nn_name+'_results')
    nn = []
    for cam in camlist:
        
        logger.debug('processing camera %s for %s detector',cam,nn_name)

        nn.append(NN.load_from_folder(os.path.join(nn_root, 'machine_'+(str(cam+1))),nn_name))

    return nn

def get_nn_cam_params(data_root,nnList,start_in_cam1=0,n_files='all'):
    
    
    n_cams = len(os.listdir(os.path.join(data_root,'data')))
    cams = process_cameras(data_root,n_cams)
    NNs= []
    for nn in nnList: 
        NNs.append(process_NN(data_root,nn,n_cams))

 
    if n_files == 'all':
        timestamps = sorted(NNs[0][0].timestamps)[start_in_cam1:] 
    else:
        timestamps = sorted(NNs[0][0].timestamps)[start_in_cam1:start_in_cam1+n_files]

    J_names = {
        0: 'Pelvis',
 
        1: 'L_Hip',
        4: 'L_Knee',
        7: 'L_Ankle',
        10: 'L_Foot',
    
        2: 'R_Hip',
        5: 'R_Knee',
        8: 'R_Ankle',
        11: 'R_Foot',
    
        3: 'Spine1',
        6: 'Spine2',
        9: 'Spine3',
        12: 'Neck',
        15: 'Head',
    
        13: 'L_Collar',
        16: 'L_Shoulder',
        18: 'L_Elbow',
        20: 'L_Wrist',
        22: 'L_Hand',
        14: 'R_Collar',
        17: 'R_Shoulder',
        19: 'R_Elbow',
        21: 'R_Wrist',
        23: 'R_Hand',
    }

    # import ipdb; ipdb.set_trace()

    intrinsics = np.zeros([n_cams,len(nn_list),3,3])
    extrinsics = np.zeros([n_cams,len(nn_list),4,4])
    covs = np.zeros([n_cams,len(nn_list),6,6])
    joints2D = np.zeros([n_cams,len(nn_list),14,2])
    probs = np.zeros([n_cams,len(nn_list),14])
    cov_j2d = np.zeros([n_cams,len(nn_list),14,2,2])

    for i in timestamps:

        # logger.debug('processing timestamp %s',NNs[0][0].tstamps_raw[i])
        logger.debug('processing timestamp %s',i)
    
        intrinsics[:] = 0
        extrinsics[:] = 0 
        covs[:] = 0
        joints2D[:] = 0
        cov_j2d[:] = 0

        cam = 0
        for nn in range(len(nn_list)):
            intrinsics[cam,nn,:,:] = cams[cam].get_intrinsic(i)
            extrinsics[cam,nn,:,:],covs[cam,nn,:,:] = cams[cam].get_extrinsic_and_cov(i)
            joints2D[cam,nn,:,:], probs[cam,nn,:] = NNs[nn][cam].get_2d_joints_and_probs(i,roi_path = cams[cam].roi[i])
            cov_j2d[cam,nn,:,:,:] = NNs[nn][cam].get_cov_j2d()
        # t = [NNs[nn][cam].tstamps_raw[i]]
        t = [i]

        for cam in range(1,n_cams):
            for nn in range(len(nn_list)):    
                j = cams[cam].get_closest_time_stamp(i)
                intrinsics[cam,nn,:,:] = cams[cam].get_intrinsic(j)
                extrinsics[cam,nn,:,:],covs[cam,nn,:,:] = cams[cam].get_extrinsic_and_cov(j)
                joints2D[cam,nn,:,:], probs[cam,nn,:] = NNs[nn][cam].get_2d_joints_and_probs(j,roi_path = cams[cam].roi[j])
                cov_j2d[cam,nn,:,:,:] = NNs[nn][cam].get_cov_j2d()
            
            # t.append(NNs[nn][cam].tstamps_raw[j])
            t.append(j)

    return intrinsics,extrinsics,covs,joints2D,cov_j2d


def processCamsNNs(data_root,nnList,camlist=None):

    assert type(nnList) == list
    assert type(nnList[0]) == str

    if camlist is None:
        n_cams = len(os.listdir(os.path.join(data_root,'data')))
    else:
        n_cams = len(camlist)
        
    n_NNs = len(nnList)
    # import ipdb; ipdb.set_trace()
    cams = process_cameras(data_root,camlist)
    NNs= []
    for nn in nnList: 
        NNs.append(process_NN(data_root,nn,camlist))
    
    
    tstamps = []
    camidx = []
    for cam in range(n_cams):
        # import ipdb;ipdb.set_trace()
        # tstamps += NNs[0][cam].timestamps
        # camidx += (cam + np.zeros(len(NNs[0][cam].timestamps),dtype=int)).tolist()
        tstamps += cams[cam].timestamps
        camidx += (cam + np.zeros(len(cams[cam].timestamps),dtype=int)).tolist()
    
    tstamp2cam = zip(tstamps,camidx)
    tstamp2cam = sorted(tstamp2cam, key = lambda x: int(x[0]))

    return n_cams,n_NNs,cams,NNs,tstamp2cam
