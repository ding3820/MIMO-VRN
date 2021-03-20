import logging
import math

from models.modules.Inv_arch import *
from models.modules.Subnet_constructor import subnet

logger = logging.getLogger('base')


####################
# define network
####################
def define_G(opt):
	opt_net = opt['network_G']
	which_model = opt_net['which_model_G']
	subnet_type = which_model['subnet_type']
	opt_datasets = opt['datasets']

	if opt_net['init']:
		init = opt_net['init']
	else:
		init = 'xavier'

	down_num = int(math.log(opt_net['scale'], 2))

	netG = Net(opt, subnet(subnet_type, init), down_num)

	return netG
