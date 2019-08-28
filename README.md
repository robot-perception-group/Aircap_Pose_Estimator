# Aircap_Pose_Estimator

![Alt](https://github.com/robot-perception-group/Aircap_Pose_Estimator/blob/master/repo_assets/teaser.png)

Get this repo:
`git clone https://github.com/robot-perception-group/Aircap_Pose_Estimator.git`

### Data
###  Coming soon. Please visit this readme again in a few days.

### Install Requirements 
- Python2.7
- ROS Melodic [http://wiki.ros.org/melodic]
- Other requirementas
	`pip install -r requirements.txt`
- torchgeometry v0.1.0
	```
	git clone https://github.com/arraiyopensource/kornia.git
	cd kornia
	git checkout v0.1.0
	python setup.py install
	```
- Download SMPL from [http://smpl.is.tue.mpg.de/] and extract its content in the parent directory _i.e._ __Aircap_Pose_Estimator/__
	
	#### optional requirements 
	- Mayavi for results visualization. Install Mayavi for Python2.7 from https://docs.enthought.com/mayavi/mayavi/installation.html#installing-with-pip


### Run Aircap_Pose_Estimator demo
- run Aircap_Pose_Estimator demo as in paper
	`python fittingscript.py /path/to/result/directory`
- results will be saved in **/path/to/result/directory**. Mean error for each joint will be in the file **/path/to/result/directory/final_err_res.npy**
- To visualize results, execute `python viz_res.py /path/to/result/directory` to launch the visualization of results alongwith the ground truth.

### Change the Aircap_Pose_Estimator parameters 
### ****************************************TODO
