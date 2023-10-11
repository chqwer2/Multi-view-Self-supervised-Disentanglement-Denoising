import argparse, os
from options import utils_option

# ----------------------------
# Input Parser
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--opt', type=str, default='', help='Path to option JSON file.')
parser.add_argument('--gpuid', type=str, default='0')

parser.add_argument('--dist', default=False)
parser.add_argument('--task', default="denoising")

parser.add_argument('--train', action='store_true')
parser.add_argument('--test', action='store_true')

parser.add_argument('--pretrain_dir', default=None)
parser.add_argument('--init_iter', type=int, default=0)


# Get Option File
opt = utils_option.parse(parser.parse_args().opt, is_train=True)


# ----------------------------
# Get Model Config
# ----------------------------
if opt["model"] == "swinv2":
    model_file = "options/model/default_swirv2.json"



model_opt = utils_option.parse(model_file, is_train=True)
opt.update(model_opt)

# ----------------------------
# Get GPU Config
# ----------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = parser.parse_args().gpuid
CUDA_VISIBLE_DEVICES = parser.parse_args().gpuid




