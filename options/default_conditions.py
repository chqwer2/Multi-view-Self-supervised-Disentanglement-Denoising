from options import utils_option
import os

# ----------------------------
# Setup Experiment Name
# ----------------------------
def setup_expname(opt):
    name = opt["train_type"]
    name += '_' + opt['model']
    # name += '_' + f"layer{len(opt['netG']['depths'])}"

    name += '_' + f"depth{opt['netG']['depths']}"
    name += '_' + f"head{opt['netG']['num_heads']}"

    name += '_' + f"lr{opt['train']['G_optimizer_lr']}"

    name += "_" + f"sigma{opt['datasets']['sigma']}"
    name += "_" + f"res{opt['datasets']['H_size']}"
    name += "_" + f"mixup{opt['datasets']['use_mixup']}"

    name += "_" + f"{opt['train']['G_optimizer_type']}"

    if opt["med"]["fused_forward"]:
        name += "_" + f"fused{opt['med']['fused_weight']}"

    name += '_' + opt['name']
    return name.rstrip("_").replace(" ", "")


def default_condition(opt):
    # ----------------------------------------
    # MeDIA
    # ----------------------------------------
    if "med" not in opt:
        opt['med'] = {}
        opt['med']['fused_weight"'] = 0.1

    if "fused_forward" not in opt["med"]:  opt["med"]["fused_forward"] = False


    if "multi_model"    not in opt:    opt["multi_model"]  = False
    if "resume"         not in opt:    opt["resume"]       = False
    if "residual"       not in opt:    opt["residual"]     = False
    if "upscale"        not in opt:    opt["upscale"]      = False

    return opt



# ----------------------------
# Load Model Function
# ----------------------------

def parser_pretrain(opt):
    # ----------------------------------------
    # Pre-Train
    # ----------------------------------------
    if opt['pretrain_dir']:
        dir = opt['pretrain_dir']
        print("Load from pretrained dir:", dir)
        init_iter_G, init_path_G = utils_option.find_last_checkpoint(dir, net_type='G')
        init_iter_E, init_path_E = utils_option.find_last_checkpoint(dir, net_type='E')
        init_iter_optimizerG, init_path_optimizerG = utils_option.find_last_checkpoint(dir, net_type='optimizerG')
        current_step = 0
    else:
        print("Load from resumed ckp pth:", opt["ckp_pth"])
        init_iter_G, init_path_G = utils_option.find_last_checkpoint(opt["ckp_pth"], net_type='G')
        init_iter_E, init_path_E = utils_option.find_last_checkpoint(opt["ckp_pth"], net_type='E')
        init_iter_optimizerG, init_path_optimizerG = utils_option.find_last_checkpoint(opt["ckp_pth"], net_type='optimizerG')
        current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)

    opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netE'] = init_path_E
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG

    opt['current_step'] = current_step
    return opt


def get_training_default(opt):
    opt['exp_base_dir'] = "../train_log"
    opt['pretrain_dir'] = None

    if opt['path']['pretrain_dir']:
        opt['pretrain_dir'] = os.path.join(opt['path']['root'], opt['path']['pretrain_dir'])

    opt = default_condition(opt)
    opt["name"] = setup_expname(opt)
    ckp_pth = os.path.join(opt['exp_base_dir'], opt["task"], opt["name"])

    opt["ckp_pth"] = ckp_pth
    opt = parser_pretrain(opt)

    os.makedirs(ckp_pth, exist_ok=True)

    utils_option.save(opt, ckp_pth)
    opt = utils_option.dict_to_nonedict(opt)
    return opt, ckp_pth