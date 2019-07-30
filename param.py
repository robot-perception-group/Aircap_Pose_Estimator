data = 'processed'        # 'raw' or 'processed'
nn_list = ['alphapose','openpose']
camlist = [0,1,2]
posePrior = 'vposer'

# idxWin is [start, stop]
idxWin = [150,9000]

# lossW is [loss2dW, lossBetaW, lossPosePrior]
lossW = [10, 0.1, 1,10,10]
optimName = 'Adam'
optimiters = [1000,100]
optim_param = [0.25,0.15]
gemanSigma = 40
camopt = True
