import os, torch
from torch.nn.parallel import DataParallel, DistributedDataParallel




def get_bare_model(network):
        """
        Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        if isinstance(network, (DataParallel, DistributedDataParallel)):
            network = network.module

        return network
    
    
def save_network(save_dir, network, network_label, iter_label):
                save_filename = '{}_{}.pth'.format(iter_label, network_label)
                save_path = os.path.join(save_dir, save_filename)
                network = get_bare_model(network)
                state_dict = network.state_dict()
                for key, param in state_dict.items():
                    state_dict[key] = param.cpu()
                torch.save(state_dict, save_path)

