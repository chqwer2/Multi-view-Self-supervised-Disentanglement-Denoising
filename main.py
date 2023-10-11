import os, glob
from options.parser import opt, parser
from options import utils_option

from models.select_network import define_network, define_multi_network
from data.select_dataset import define_dataset
from lib.utils_train import *
from options.default_conditions import get_training_default
import wandb
wandb.login(key="ee52f649687b804de41f2c26b2049c7cd3e4db99")



def main(opt, args, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        
    # ----------------------------
    # Test
    # ----------------------------
    if args.test:
        home = opt["pretrain"]["home"]
        all = opt["pretrain"]["all"]
        opt = utils_option.dict_to_nonedict(opt)

        if all:
            print("read dir from all")
            experiment_dirs = glob.glob(os.path.join(home, '*'))
        else:
            experiment_dirs = []
            for i in opt["pretrain"]["results"]:
                experiment_dirs.append(os.path.join(home, i[1]))

        failed = []
        test_types = opt["datasets"]["test_types"]  # gaussian
        for noise_type in test_types:

            print("[Testing] on:", noise_type)

            for experiment in experiment_dirs:

                opt_test = utils_option.parse(os.path.join(experiment, "option.json"), is_train=True)

                opt['pretrain_dir'] = experiment
                opt["init_iter"] = 0

                net = define_network(opt_test, residual=opt_test["residual"]).to(device)

                print("Load from:", experiment)

                # If failed, uncomment the following line
                # experiment = glob.escape(experiment)
                # network_test(opt, net, loadpath=experiment, noise_types=noise_type)

                try:
                    network_test(opt, net, loadpath=experiment, noise_types=noise_type)
                except:
                    failed.append(experiment)

        print("failed exp:", failed)
        print("Done and results saved in:",  experiment_dirs[0])


    # ----------------------------
    # Train
    # ----------------------------
    elif args.train:
        opt, ckp_pth = get_training_default(opt)
        model_dict = define_multi_network(opt, device)

        train_loader, val_loader, _ = define_dataset(opt, duplicate_val=True)
        
        print("Training the network...")
        network_training(opt, model_dict,
                         train_loader, val_loader, ckp_pth,
                         current_step=opt['current_step'], device=device)



if __name__ == "__main__":
    main(opt, args=parser.parse_args())
    print("Done...")









