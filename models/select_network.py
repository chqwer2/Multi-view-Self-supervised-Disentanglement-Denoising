import functools
from torch.nn import init
from optimization.optimization import Optimization
from models.network_swin_v2 import Swin_v2


"""
# --------------------------------------------
# select the network of G, D and F
# --------------------------------------------
"""

def get_model_mac(net, inp_shape=(3, 256, 256)):
    # pip install ptflops
    from ptflops import get_model_complexity_info
    FLOPS = 0
    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=True)

    # params = float(params[:-4])
    # MACs (G) in log scale
    print(params)
    macs = float(macs[:-4]) + FLOPS / 10 ** 9

    print('mac', macs, params)


def weights_init():
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__

            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')

            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun

# --------------------------------------------
# Generator, netG, G
# --------------------------------------------
def define_network(opt, residual=False):
    opt_net = opt['netG']
    net_type = opt_net['net_type']

    print("Selected net type:", net_type)

    # ----------------------------------------
    # Swin_v2
    # ----------------------------------------
    if net_type == 'swin_v2':
        net = Swin_v2
        netG = net(upscale=opt['upscale'],
                   in_chans=opt_net['in_chans'],
                   img_size=opt["datasets"]["H_size"],
                   window_size=opt_net['window_size'],
                   img_range=opt_net['img_range'],
                   depths=opt_net['depths'],
                   embed_dim=opt_net['embed_dim'],
                   num_heads=opt_net['num_heads'],
                   mlp_ratio=opt_net['mlp_ratio'],
                   upsampler=opt_net['upsampler'],
                   resi_connection=opt_net['resi_connection'],
                   residual=residual)

        # MAC = get_model_mac(netG.cuda(),
        #               inp_shape=(3, opt["datasets"]["H_size"],
        #                          opt["datasets"]["H_size"]))


        # ----------------------------------------
        # initialize weights
        # ----------------------------------------
        netG.apply(weights_init())
        
        return netG
    
    else:
        raise NotImplementedError('netG [{:s}] is not found.'.format(net_type))


def define_multi_network(opt, device):

    # Initialized Network
    model_dict = {}

    model_dict['net'] = define_network(opt, residual=opt["residual"]).to(device)
    model_dict['trainer'] = Optimization(opt, model_dict['net'], optname="net")

    if opt["multi_model"]:
        model_dict['trainer'].netN = define_network(opt, residual=True).to(device)
        model_dict['trainer'].netD = define_network(opt).to(device)  # , decoder_only=True



    model_dict['trainer'].get_optimizer()


    return model_dict

def init_weights(net, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
    """
    # Kai Zhang, https://github.com/cszn/KAIR
    #
    # Args:
    #   init_type:
    #       default, none: pass init_weights
    #       normal; normal; xavier_normal; xavier_uniform;
    #       kaiming_normal; kaiming_uniform; orthogonal
    #   init_bn_type:
    #       uniform; constant
    #   gain:
    #       0.2
    """

    def init_fn(m, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
        classname = m.__class__.__name__

        if classname.find('Conv') != -1 or classname.find('Linear') != -1:

            if init_type == 'normal':
                init.normal_(m.weight.data, 0, 0.1)
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'uniform':
                init.uniform_(m.weight.data, -0.2, 0.2)
                m.weight.data.mul_(gain)

            elif init_type == 'xavier_normal':
                init.xavier_normal_(m.weight.data, gain=gain)
                m.weight.data.clamp_(-1, 1)

            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=gain)

            elif init_type == 'kaiming_normal':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'kaiming_uniform':
                init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.mul_(gain)

            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)

            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_type))

            if m.bias is not None:
                m.bias.data.zero_()

        elif classname.find('BatchNorm2d') != -1:

            if init_bn_type == 'uniform':  # preferred
                if m.affine:
                    init.uniform_(m.weight.data, 0.1, 1.0)
                    init.constant_(m.bias.data, 0.0)
            elif init_bn_type == 'constant':
                if m.affine:
                    init.constant_(m.weight.data, 1.0)
                    init.constant_(m.bias.data, 0.0)
            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_bn_type))

    if init_type not in ['default', 'none']:
        print('Initialization method [{:s} + {:s}], gain is [{:.2f}]'.format(init_type, init_bn_type, gain))
        fn = functools.partial(init_fn, init_type=init_type, init_bn_type=init_bn_type, gain=gain)
        net.apply(fn)
    else:
        print('Pass this initialization! Initialization was done during network definition!')
