import torch, copy
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.optim import Adam, AdamW, SGD
from data.noise2void.masker import generate_mask as n2v_generate_mask
from lib.utils_regularizers import regularizer_orth, regularizer_clip
from optimization.model_base import BaseOptimizer



class Optimization(BaseOptimizer):
    def __init__(self, opt, net, optname):
        super(Optimization, self).__init__(opt, optname)
        
        self.opt = opt
        self.opt_train = self.opt["train"]
        self.netG = net

        if self.opt_train['E_decay'] > 0:
            self.netE = copy.deepcopy(net).eval()
            self.netE.requires_grad = False

        self.define_loss()

        # self.define_optimizer(self.netG)
        try:
            self.load()
        except:
            print("Loading pretrain failed")
            
        if self.opt["train_type"] == "n2s":
            from data.noise2self.mask import Masker
            self.n2s_masker = Masker(width=4, mode='interpolate')

    def load(self):
        self.load_nets()
        self.load_optimizer()  # TODO


    # ----------------------------------------
    # define optimizer
    # ----------------------------------------
    def get_optimizer(self):

                
        if self.opt_train['G_optimizer_type'] == 'adam':
            optimizer = Adam
        elif self.opt_train['G_optimizer_type'] == 'adamw':
            optimizer = AdamW
        elif self.opt_train['G_optimizer_type'] == 'sgd':
            optimizer = SGD

        G_optim_params = list(self.netG.parameters())
        if self.opt['multi_model']:

            G_optim_params += list(self.netN.parameters())
            G_optim_params += list(self.netD.parameters())

        self.G_optimizer = optimizer(
                                [{"params": G_optim_params, 
                                "lr":self.opt_train['G_optimizer_lr'],  
                                "betas": self.opt_train['G_optimizer_betas'],
                                "weight_decay":self.opt_train['G_optimizer_wd']}]
                                )

        self.scheduler = self.define_scheduler(self.G_optimizer)



    def set_optimizer(self, G_optimizer, scheduler):
        self.G_optimizer = G_optimizer
        self.scheduler = scheduler


    # ----------------------------------------
    # define scheduler, only "MultiStepLR"
    # ----------------------------------------
    def define_scheduler(self, G_optimizer):
        if self.opt["train"]['G_scheduler_type'] == 'MultiStepLR':
            scheduler = lr_scheduler.MultiStepLR(G_optimizer,
                                                            self.opt_train['G_scheduler_milestones'],
                                                            self.opt_train['G_scheduler_gamma'])

        elif self.opt_train['G_scheduler_type'] == 'CosineAnnealingWarmRestarts':
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(G_optimizer,
                                                            self.opt_train['G_scheduler_periods'],
                                                            self.opt_train['G_scheduler_restart_weights'],
                                                            self.opt_train['G_scheduler_eta_min'])
        else:
            raise NotImplementedError

        return scheduler

    def update_learning_rate(self, n):
        self.scheduler.step(n)
            
    def update_optimizer(self, n, loss, model_type="G"):

        self.G_optimizer.zero_grad()
        loss.backward()

        # ------------------------------------
        # clip_grad
        # ------------------------------------
        # `clip_grad_norm` helps prevent the exploding gradient problem.
        G_optimizer_clipgrad = self.opt_train['G_optimizer_clipgrad'] if self.opt_train['G_optimizer_clipgrad'] else 0
        if G_optimizer_clipgrad > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.opt_train['G_optimizer_clipgrad'],
                                           norm_type=2)

        self.G_optimizer.step()

        # ------------------------------------
        # regularizer
        # ------------------------------------
        G_regularizer_orthstep = self.opt_train['G_regularizer_orthstep'] if self.opt_train['G_regularizer_orthstep'] else 0
        if G_regularizer_orthstep > 0 and n % G_regularizer_orthstep == 0 and n % self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_orth)
        G_regularizer_clipstep = self.opt_train['G_regularizer_clipstep'] if self.opt_train['G_regularizer_clipstep'] else 0
        if G_regularizer_clipstep > 0 and n % G_regularizer_clipstep == 0 and n % self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_clip)


        self.update_learning_rate(n)
        if self.opt_train['E_decay'] > 0 and model_type == "G":
            self.update_E(self.opt_train['E_decay'])

   
    
    def forward(self, n, net_input, noisy_target=None, target=None,
                train_type="n2c", downsample=False, model_type="G"):
        if model_type == "G":
            net = self.netG
        elif model_type == "N" and self.opt["multi_model"]:
            net = self.netN

        net.train()

        if train_type == "n2s":
            net_input, mask = self.n2s_masker.mask(net_input, n)
            net_input, mask = net_input.detach(), mask.detach()

        elif train_type == "n2v":
            target = net_input.clone()
            net_input, _, mask = n2v_generate_mask(net_input)
            mask = torch.from_numpy(mask).cuda().detach()

            # img_H[negmask] = 1.0
            #  https://github.com/juglab/n2v/blob/fc6bf3fd8974fc072272ced97a5981d7406c04f9/n2v/internals/N2V_DataWrapper.py#L6

        net_output = net(net_input, downsample=downsample)

        if train_type == "n2c" or train_type == "r2r":
            g_loss = self.lossfn(net_output, target)

        elif train_type == "n2n":
            g_loss = self.lossfn(net_output, noisy_target)

        elif train_type == "n2s" or train_type == "n2v":
            g_loss = self.lossfn(net_output * mask, noisy_target * mask)

        elif train_type == "med":
            g_loss = self.lossfn(net_output, noisy_target)

        elif train_type == "med_fused":
            g_loss = self.lossfn(net_output, noisy_target)
        
        self.update_optimizer(n, g_loss, model_type=model_type)

        # ------------------------------------
        # regularizer
        # ------------------------------------
        # G_regularizer_orthstep = self.opt_train['G_regularizer_orthstep'] if self.opt_train[
        #     'G_regularizer_orthstep'] else 0
        # if G_regularizer_orthstep > 0 and current_step % G_regularizer_orthstep == 0 and current_step % \
        #         self.opt['train']['checkpoint_save'] != 0:
        #     self.netG.apply(regularizer_orth)
        # G_regularizer_clipstep = self.opt_train['G_regularizer_clipstep'] if self.opt_train[
        #     'G_regularizer_clipstep'] else 0
        # if G_regularizer_clipstep > 0 and current_step % G_regularizer_clipstep == 0 and current_step % \
        #         self.opt['train']['checkpoint_save'] != 0:
        #     self.netG.apply(regularizer_clip)
        # self.log_dict['G_loss'] = G_loss.item()/self.E.size()[0]  # if `reduction='sum'`
        # self.log_dict['G_loss'] = G_loss.item()


        # if self.opt_train['E_decay'] > 0:
        #     self.update_E(self.opt_train['E_decay'])

        return g_loss

    def fused_forward(self, n, net_input_L1, net_input_L2, data=None):
        self.netG.train()
        net_feature_L1 = self.netG.get_feature(net_input_L1)
        net_feature_L2 = self.netG.get_feature(net_input_L2)

        # Manifold Bernoulli
        if self.opt['med']["fused_method"] == "bernoulli": 
            ber_x_mask = (torch.rand(net_feature_L1.shape) > 0.5).type(torch.uint8).cuda().detach()
            ber_L_mask = (torch.rand(net_input_L1.shape) > 0.5).type(torch.uint8).cuda().detach() 
            
            mix_feature = ber_x_mask * net_feature_L1 + (1-ber_x_mask) * net_feature_L2  
            mix_input = ber_L_mask * net_input_L1 + (1-ber_L_mask) * net_input_L2

        # Manifold Mixup
        elif self.opt['med']["fused_method"] == "mixup":
            mixup = torch.rand(1).cuda()
            mix_feature = mixup * net_feature_L1 + (1-mixup) * net_feature_L2  
            mix_input = mixup * net_input_L1 + (1-mixup) * net_input_L2

        mix_out = self.netG.forward_feature(mix_feature)
        g_loss = self.lossfn( mix_out, mix_input.detach()) * self.opt["med"]["fused_weight"]

        self.update_optimizer(n, g_loss)
        return g_loss



    def forward_disentangle(self, n, net_input_L1, net_input_L2):
        self.netG.train()
        self.netN.train()
        self.netD.train()

        img_rep_L1 = self.netG.get_feature(net_input_L1)
        noise_rep_L2, L2_mean, L2_size, H, W = self.netN.get_feature(net_input_L2, return_mean=True)

        # Add
        combined_rep = img_rep_L1 + noise_rep_L2
        net_output = self.netD.forward_feature(combined_rep,
                                          input_mean=L2_mean,
                                          input_x_size=L2_size, H=H, W=W)

        g_loss = self.lossfn(net_output, net_input_L2) * 2 * self.opt["med"]["fused_weight"]

        self.update_optimizer(n, g_loss, model_type="D")

        return g_loss



