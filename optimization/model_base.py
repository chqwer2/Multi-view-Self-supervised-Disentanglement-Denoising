import os
import torch
import torch.nn as nn
from lib.utils_bnorm import merge_bn, tidy_sequential
from torch.nn.parallel import DataParallel, DistributedDataParallel
from optimization.losses import CharbonnierLoss, UnL2, SSIMLoss
from optimization.losses import GANLoss, PerceptualLoss
from collections import OrderedDict


class BaseOptimizer(nn.Module):
    def __init__(self, opt, optname=None):
        super().__init__()
        self.opt = opt  # opt
        self.save_dir = os.path.join(opt['ckp_pth'], optname)  # save models

        self.device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.is_train = opt['is_train']  # training or not
        self.schedulers = []  # schedulers
        # self.freeze_encoder = opt['path']['models']

    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """

    def init_train(self):
        pass

    def load(self):
        pass

        # ----------------------------------------
        # save model / optimizer(optional)
        # ----------------------------------------

    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, 'G', iter_label)

        if self.opt_train['E_decay'] > 0:
            self.save_network(self.save_dir, self.netE, 'E', iter_label)

        if self.opt['multi_model']:
            self.save_network(self.save_dir, self.netD, 'D', iter_label)
            self.save_network(self.save_dir, self.netN, 'N', iter_label)

        if self.opt_train['G_optimizer_reuse']:
            self.save_optimizer(self.save_dir,
                                self.G_optimizer,
                                'optimizerG', iter_label)

    # ----------------------------------------
    # define loss
    # ----------------------------------------
    def define_loss(self, reduction='mean'):
        G_lossfn_type = self.opt_train['G_lossfn_type']
        dtype = torch.cuda.FloatTensor

        if G_lossfn_type == 'l1':
            self.lossfn = nn.L1Loss().type(dtype).to(self.device)

        elif G_lossfn_type == 'l2':
            self.lossfn = nn.MSELoss().type(dtype).to(self.device)

        elif G_lossfn_type == 'l2sum':
            self.lossfn = nn.MSELoss(reduction='sum').to(self.device)

        elif G_lossfn_type == 'ssim':
            self.lossfn = SSIMLoss().type(dtype).to(self.device)

        elif G_lossfn_type == 'charbonnier':
            self.lossfn = CharbonnierLoss(self.opt_train['G_charbonnier_eps']).to(self.device)

        elif G_lossfn_type == 'charbonnier_sum':
            self.lossfn = CharbonnierLoss(self.opt_train['G_charbonnier_eps'], reduction='sum').to(self.device)


        elif G_lossfn_type == "charbonnier_with_blur":
            self.lossfn = UnL2(self.opt_train['G_charbonnier_eps']).to(self.device)

        else:
            raise NotImplementedError('Loss type [{:s}] is not found.'.format(G_lossfn_type))

        self.lossfn_weight = self.opt_train['G_lossfn_weight']
        # self.lossfn_fused = UnL2().type(dtype)
        self.lossfn_fused = nn.L1Loss().type(dtype).to(self.device)  # UnL2().type(dtype)

    def define_optimizer(self):
        pass

    def define_scheduler(self):
        pass

    """
    # ----------------------------------------
    # Optimization during training with data
    # Testing/evaluation
    # ----------------------------------------
    """

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def current_visuals(self):
        pass

    def current_losses(self):
        pass

    def update_learning_rate(self, n):
        for scheduler in self.schedulers:
            scheduler.step(n)

    def current_learning_rate(self):
        return self.schedulers[0].get_lr()[0]

    def requires_grad(self, model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    """
    # ----------------------------------------
    # Information of net
    # ----------------------------------------
    """

    def print_network(self):
        pass

    def info_network(self):
        pass

    def print_params(self):
        pass

    def info_params(self):
        pass

    def get_bare_model(self, network):
        """Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        if isinstance(network, (DataParallel, DistributedDataParallel)):
            network = network.module
        return network

    def model_to_device(self, network):
        """Model to device. It also warps models with DistributedDataParallel
        or DataParallel.
        Args:
            network (nn.Module)
        """
        network = network.to(self.device)
        if self.opt['dist']:
            find_unused_parameters = self.opt.get('find_unused_parameters', True)
            use_static_graph = self.opt.get('use_static_graph', False)
            network = DistributedDataParallel(network, device_ids=[torch.cuda.current_device()],
                                              find_unused_parameters=find_unused_parameters)
            if use_static_graph:
                print('Using static graph. Make sure that "unused parameters" will not change during training loop.')
                network._set_static_graph()
        else:
            network = DataParallel(network)
        return network

    # ----------------------------------------
    # network name and number of parameters
    # ----------------------------------------
    def describe_network(self, network):
        network = self.get_bare_model(network)
        msg = '\n'
        msg += 'Networks name: {}'.format(network.__class__.__name__) + '\n'
        msg += 'Params number: {}'.format(sum(map(lambda x: x.numel(), network.parameters()))) + '\n'
        msg += 'Net structure:\n{}'.format(str(network)) + '\n'
        return msg

    # ----------------------------------------
    # parameters description
    # ----------------------------------------
    def describe_params(self, network):
        network = self.get_bare_model(network)
        msg = '\n'
        msg += ' | {:^6s} | {:^6s} | {:^6s} | {:^6s} || {:<20s}'.format('mean', 'min', 'max', 'std', 'shape',
                                                                        'param_name') + '\n'
        for name, param in network.state_dict().items():
            if not 'num_batches_tracked' in name:
                v = param.data.clone().float()
                msg += ' | {:>6.3f} | {:>6.3f} | {:>6.3f} | {:>6.3f} | {} || {:s}'.format(v.mean(), v.min(), v.max(),
                                                                                          v.std(), v.shape, name) + '\n'
        return msg

    """
    # ----------------------------------------
    # Save prameters
    # Load prameters
    # ----------------------------------------
    """

    # ----------------------------------------
    # save the state_dict of the network
    # ----------------------------------------
    def save_network(self, save_dir, network, network_label, iter_label):
        save_filename = '{}_{}.pth'.format(iter_label, network_label)
        save_path = os.path.join(save_dir, network_label)

        os.makedirs(save_path, exist_ok=True)

        save_path = os.path.join(save_path, save_filename)

        network = self.get_bare_model(network)
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    # ----------------------------------------
    # load the state_dict of the network
    # ----------------------------------------
    def load_network(self, load_path, network, strict=True, param_key='params'):
        network = self.get_bare_model(network)
        if strict:
            state_dict = torch.load(load_path)
            if param_key in state_dict.keys():
                state_dict = state_dict[param_key]
            network.load_state_dict(state_dict, strict=strict)
        else:
            state_dict_old = torch.load(load_path)
            if param_key in state_dict_old.keys():
                state_dict_old = state_dict_old[param_key]
            state_dict = network.state_dict()
            for ((key_old, param_old), (key, param)) in zip(state_dict_old.items(), state_dict.items()):
                state_dict[key] = param_old
            network.load_state_dict(state_dict, strict=True)
            del state_dict_old, state_dict

    # ----------------------------------------
    # save the state_dict of the optimizer
    # ----------------------------------------
    def save_optimizer(self, save_dir, optimizer, optimizer_label, iter_label):
        save_filename = '{}_{}.pth'.format(iter_label, optimizer_label)

        save_path = os.path.join(save_dir, optimizer_label)

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        save_path = os.path.join(save_path, save_filename)

        torch.save(optimizer.state_dict(), save_path)

    # ----------------------------------------
    # load pre-trained G model
    # ----------------------------------------
    def load_nets(self):
        load_path_G = self.opt['path']['pretrained_netG']
        if load_path_G is not None:
            print('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, strict=self.opt_train['G_param_strict'], param_key='params')

        load_path_E = self.opt['path']['pretrained_netE']
        if self.opt_train['E_decay'] > 0:
            if load_path_E is not None:
                print('Loading model for E [{:s}] ...'.format(load_path_E))
                self.load_network(load_path_E, self.netE, strict=self.opt_train['E_param_strict'],
                                  param_key='params_ema')
            else:
                print('Copying model for E ...')
                self.update_E(0)
            self.netE.eval()

    # ----------------------------------------
    # load the state_dict of the optimizer
    # ----------------------------------------
    def load_optimizer(self):
        load_path_G = self.opt['path']['pretrained_optimizerG']

        self.G_optimizer.load_state_dict(
            torch.load(load_path_G, map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device())))

    # def update_E(self, decay=0.999):
    #     netG = self.get_bare_model(self.netG)
    #     netG_params = dict(netG.named_parameters())
    #     netE_params = dict(self.netE.named_parameters())
    #     for k in netG_params.keys():
    #         netE_params[k].data.mul_(decay).add_(netG_params[k].data, alpha=1-decay)

    def update_E(self, decay=0.999):

        model_params = OrderedDict(self.netG.named_parameters())
        shadow_params = OrderedDict(self.netE.named_parameters())
        for name, param in model_params.items():
            # param.requires_grad = False
            shadow_params[name].requires_grad = False
            # see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
            shadow_params[name].sub_((1. - decay) * (shadow_params[name] - param.detach()))



        model_buffers = OrderedDict(self.netG.named_buffers())
        shadow_buffers = OrderedDict(self.netE.named_buffers())

        # check if both model contains the same set of keys
        assert model_buffers.keys() == shadow_buffers.keys()

        for name, buffer in model_buffers.items():
            shadow_buffers[name].copy_(buffer)

    """
    # ----------------------------------------
    # Merge Batch Normalization for training
    # Merge Batch Normalization for testing
    # ----------------------------------------
    """

    # ----------------------------------------
    # merge bn during training
    # ----------------------------------------
    def merge_bnorm_train(self):
        merge_bn(self.netG)
        tidy_sequential(self.netG)
        self.define_optimizer()
        self.define_scheduler()

    # ----------------------------------------
    # merge bn before testing
    # ----------------------------------------
    def merge_bnorm_test(self):
        merge_bn(self.netG)
        tidy_sequential(self.netG)