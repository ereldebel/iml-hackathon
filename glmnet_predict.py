import sys
sys.path.append('../test')
sys.path.append('../lib')
import scipy, importlib, pprint, matplotlib.pyplot as plt, warnings
from glmnet_python import glmnet; from glmnetPlot import glmnetPlot
from glmnetPrint import glmnetPrint; from glmnetCoef import glmnetCoef; from glmnetPredict import glmnetPredict
from cvglmnet import cvglmnet; from cvglmnetCoef import cvglmnetCoef
from cvglmnetPlot import cvglmnetPlot; from cvglmnetPredict import cvglmnetPredict

if __name__ == '__main__':
	# parameters
	baseDataDir= '../data/'

	# load data
	x = scipy.loadtxt(baseDataDir + 'QuickStartExampleX.dat', dtype = scipy.float64)
	y = scipy.loadtxt(baseDataDir + 'QuickStartExampleY.dat', dtype = scipy.float64)

	# create weights
	t = scipy.ones((50, 1), dtype = scipy.float64)
	wts = scipy.row_stack((t, 2*t))

	# call glmnet
	fit = glmnet(x = x.copy(), y = y.copy(), family = 'gaussian', \
	                    weights = wts, \
	                    alpha = 0.2, nlambda = 20
	                    )

	glmnetPrint(fit)