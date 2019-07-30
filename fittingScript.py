from __future__ import print_function, division
import sys, os
import param as config
import aircap_main
import json


outDirName = sys.argv[1]

if config.data == 'raw':
    data_root = './data/raw'
elif config.data == 'processed':
    data_root = './data/processed'
else:
	sys.exit("data parameter should be either 'raw' or 'processed'")

os.mkdir(outDirName)

fitter = aircap_main.aircapFitter(data_root=data_root,nn_list=config.nn_list,camlist=config.camlist)
fitter.fit(idxWin=config.idxWin,
        lossW = config.lossW,
        optimName=config.optimName,
        optim_param=config.optim_param,
        optimiters=config.optimiters,
        gemanSigma=config.gemanSigma,
        poseprior=config.posePrior,
        camopt=config.camopt)

fitter.save_results(os.path.join(outDirName,'results'))

with open(os.path.join(outDirName,'config.txt'),'w') as f:
	f.write(json.dumps({key:config.__dict__[key] for key in dir(config) if not key.startswith('__')}))

import error_script
error_script.err_func(os.path.join(outDirName,'results.npz'),config.data,outDirName)
