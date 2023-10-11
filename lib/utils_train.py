import torch, os, logging
import torch.cuda as cuda
from lib import utils_logger
from options import utils_option as option

import numpy as np
import matplotlib.pyplot as plt
import warnings
import time

# own files import
from lib.utils import batch_PSNR, batch_SSIM, output_to_image, graph_error, save_one_image
from lib.utils import save_ckp, load_ckp, base_path

from data.select_dataset import define_test_dataset
from metrics.metrics import metrics
from tqdm import tqdm
import glob



warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (16, 9)

error_plot_freq = 20
seed = 42
INT_MAX = 2147483647
error_tolerence = 2
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)



def network_training(opt, model_dict, train_loader, val_loader, ckp_pth, device, current_step=0):

    ### Preparing for training
    error_list = []
    start_epoch = 0
    best_model_saved = True
    ckp_saved = True
    previous_batch_error = INT_MAX  # initialise to a large value
    best_error = INT_MAX
    print_interval = 20  # iterations

    epochs = opt["train"]["epochs"]

    trainer = model_dict['trainer']
    net = model_dict['net']


    logger_name = 'train'
    utils_logger.logger_info(logger_name, os.path.join(ckp_pth, logger_name + '.log'))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))


    for epoch in range(start_epoch, epochs):
        batch_error = 0
        epoch_start_time = time.time()

        if opt["task"] == "denoising" or opt["task"] == "med":
            for i, data in enumerate(train_loader, 0):

                current_step += 1
                target = data["H"].to(device)
                net_input_L1 = data["L1"].to(device)
                net_input_L2 = data["L2"].to(device)

                # Main clean forward
                loss = trainer(n=current_step, net_input=net_input_L1.clone().detach(),
                                                 noisy_target=net_input_L2.clone().detach(),
                                                target=target.clone(), train_type=opt["train_type"],
                                                downsample=None, model_type= "G")

                batch_error += loss.item()

                if opt["train_type"] == "med":
                    loss = trainer(n=current_step, net_input=net_input_L2.clone().detach(),
                                     noisy_target=net_input_L1.clone().detach(),
                                     target=target.clone().detach(), train_type=opt["train_type"], downsample=None
                                     , model_type= "G")
                    batch_error += loss.item()

                if opt["med"]["fused_forward"] == True:
                    loss_fused = trainer.fused_forward(
                        n=current_step, net_input_L1=net_input_L1.clone(),
                        net_input_L2=net_input_L2.clone(), data=data)
                    batch_error += loss_fused.item()


                # forward disentanglement and noise model
                if opt["multi_model"]:

                    noise_loss = trainer(n=current_step, net_input=net_input_L1.clone().detach(),
                                                                 noisy_target=net_input_L2.clone().detach(),
                                                                 target=target.clone(), train_type=opt["train_type"],
                                                 model_type= "N")

                    dis_loss = trainer.forward_disentangle(
                            current_step, net_input_L1.clone().detach(),
                            net_input_L2.clone().detach())


                if (current_step % 200) == 0:   # print_interval
                    if opt["multi_model"]:
                        logger.info('[%d] loss: %.3f, noise loss: %.3f, disen loss: %.3f' % (epoch + 1, loss.item(),
                                                      noise_loss.item(), dis_loss.item()))  # current_step
                    else:
                        logger.info('[%d] loss: %.3f' % (epoch + 1, loss.item()))  # current_step
                    # break


        ### find one epoch training time
        one_epoch_time = time.time() - epoch_start_time
        print("One epoch time: " + str(one_epoch_time))


        ### if error is too large, roll back, otherwise save and continue
        if batch_error > error_tolerence*previous_batch_error and (best_model_saved or ckp_saved):
            if ckp_saved:
                print("Current error is too large, loading the best model")
                net, optimizer, start_epoch, error_list, best_psnr = \
                    load_ckp(ckp_pth+"/"+"best_model.pt", net, optimizer)
            else:
                raise Exception("Error is too large, but no models to load")

        else:
            if batch_error > error_tolerence*previous_batch_error:
                print("Current error is too large, but cannot roll back")
            else:
                previous_batch_error = batch_error

        error_list.append(batch_error)


        ###if error is the smallest save it as the best model
        if batch_error < best_error:
            best_error = batch_error
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'optimizer': trainer.state_dict(),
                'error_list': error_list,
                'best_error': best_error
            }
            save_ckp(checkpoint, True, os.path.join(ckp_pth, "best_model.pt"))
            best_model_saved = True

            print("New Minimum Error Recorded!")


        if ((epoch + 1) % error_plot_freq) == 0 or epoch == epochs - 1:
            graph_error(error_list[1:], "")

        if opt['datasets']["single_image"]:
            print("Epoch:", epoch)
            opt['train']['checkpoint_save'] = 100



        if (epoch + 1) % opt['train']['checkpoint_save'] == 0:

            network_validation(opt, trainer.netE,  val_loader, ckp_pth,
                               logger=logger, epoch=current_step, sr=1)

            trainer.save(current_step)
            logger.info('Saving the model...')

            if opt["multi_model"]:
                network_validation(opt, trainer.netN,
                                   val_loader, ckp_pth, logger=logger, epoch=str(current_step) + "_noise",
                                   sr=(opt['upscale'] if opt['upscale'] else 1))



        if current_step == opt["train"]["maxstep"]:
            logger.info('Training Done...')
            return 0




def network_validation(opt, net, val_dataloader, ckp_pth, logger=None, epoch=0, window_size=8, sr=1):
    psnr_list, ssim_list, deleted_indices = [], [], []

    border = sr if opt["task"] == "super_resolution" else 0  # scale factor or denoising
    patch_size = opt["datasets"]["H_size"]

    logger.info(f"Validating Step {epoch}...")

    local_window_dim = window_size

    start_time = time.time()
    ### iterate through the validation dataset
    for idx, val_data in enumerate(val_dataloader):

        # if(idx not in deleted_indices):
        # logger.info(f"Evaluating the {idx}th sample: ")
        ### pad the image to make sure H==W fits the network requirements

        net_input = val_data["L1"]

        #  (320,480,3) (321,481,3)
        # Validated

        _, _, h_old, w_old = net_input.size()

        h_original = h_old * sr
        w_original = w_old * sr
        # print("h_original, w_original", h_original, w_original)

        multiplier = max(h_old // local_window_dim + 1, w_old // local_window_dim + 1)
        h_pad = (multiplier) * local_window_dim - h_old
        w_pad = (multiplier) * local_window_dim - w_old
        net_input = torch.cat([net_input, torch.flip(net_input, [2])], 2)[:, :, :h_old + h_pad, :]
        net_input = torch.cat([net_input, torch.flip(net_input, [3])], 3)[:, :, :, :w_old + w_pad]

        # pad again if h/w or w/h ratio is bigger than 2
        if h_pad > h_old or w_pad > w_old:
            _, _, h_old, w_old = net_input.size()
            multiplier = max(h_old // local_window_dim + 1, w_old // local_window_dim + 1)
            h_pad = (multiplier) * local_window_dim - h_old
            w_pad = (multiplier) * local_window_dim - w_old
            net_input = torch.cat([net_input, torch.flip(net_input, [2])], 2)[:, :, :h_old + h_pad, :]
            net_input = torch.cat([net_input, torch.flip(net_input, [3])], 3)[:, :, :, :w_old + w_pad]

        ### evaluate
        _, _, new_h, new_w = net_input.size()
        assert new_h == new_w, "Input image should have square dimension"


        net_input = net_input.cuda()
        target = val_data["H"].cuda()

        net.eval()

        with torch.no_grad():
            net_output = test_by_tile(net_input, net, local_window_dim, sf=sr)


        ### crop the output
        net_output = net_output[:, :, :h_original, :w_original]
        output_data = net_output.cpu().detach().numpy()  # B C H W
        output_data = np.transpose(output_data, (0, 2, 3, 1))  # B H W C
        target_data = target.cpu().detach().numpy()
        target_data = np.transpose(target_data, (0, 2, 3, 1))
        input_data = val_data["H"].cpu().detach().numpy()
        input_data = np.transpose(input_data, (0, 2, 3, 1))

        ### calculate the PSNR and SSIM
        psnr = batch_PSNR(target_data, output_data, True).item()
        ssim = batch_SSIM(target_data, output_data).item()

        psnr_list.append(psnr)
        ssim_list.append(ssim)

        cuda.empty_cache()
        run_time = time.time() - start_time

        # logger.info('{:->4d}--> {:>5f}s | PSNR:{:<4.2f}dB | SSIM:{:<4.2f}'.format(idx, run_time, psnr, ssim))
        # logger.info(f"Average time {run_time}: PSNR {avg_psnr}, SSIM {avg_ssim}")

    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)

    logger.info("saved path: " + ckp_pth)
    logger.info('<epoch:{}, Average PSNR : {:<.2f}dB | Average SSIM : {:<.2f}'.format(epoch, avg_psnr, avg_ssim))


    return avg_psnr, avg_ssim




# ----------------- TEST -------------------
def network_test(opt, net, logger=None, loadpath=None, local_window_dim=8, noise_types="", sr=1):

    logger_name = 'test'

    try:
        utils_logger.logger_info(logger_name, os.path.join(loadpath, logger_name + '.log'), mode='a')  #
    except:
        utils_logger.logger_info(logger_name, os.path.join(glob.escape(loadpath), logger_name + '.log'), mode='a')

    logger = logging.getLogger(logger_name)
    compute_metrics = metrics()

    if loadpath:
        model_type = "E"
        netname = "net"
        init_iter, init_path = option.find_last_checkpoint(os.path.join(loadpath, netname),
                                                           net_type=model_type)
        print("found iter:", init_iter)
        if opt["init_iter"] > 1000:
            init_iter = opt["init_iter"]

        model_path = '{}_{}.pth'.format(init_iter, model_type)
        model_path = os.path.join(os.path.join(loadpath, netname), model_type, model_path)

        state_dict = torch.load(model_path)
        param_key = 'params'
        if param_key in state_dict.keys():
            state_dict = state_dict[param_key]

        net.load_state_dict(state_dict, strict=True)
        logger.info("Testing load path from: " + loadpath)

    else:
        print("You need to input a loadpath")

    model_name = loadpath.split("/")[-1]

    test_datasets = opt["datasets"]["test_datasets"]  # datasets

    # Recursive Datasets
    for dataset in test_datasets:
        result_dir = os.path.join(loadpath, "test_results", dataset)

        print(f"Tesing the Datsets: " + dataset)

        n = "_".join(noise_types)
        noise_dir = os.path.join(result_dir, n)

        os.makedirs(noise_dir, exist_ok=True)

        psnr_list, ssim_list, deleted_indices = [], [], []
        metric_feature = []

        start_time = time.time()
        ### iterate through the validation dataset
        test_dataloader = define_test_dataset(opt=opt, datasets=dataset, noise_types=noise_types)

        print("Testing model:", opt['pretrain_dir'])
        print(f"Tesing the Type: " + n)

        for idx, val_data in tqdm(enumerate(test_dataloader)):

            target_data, output_data, input_data = input_single_image(val_data, net, local_window_dim,
                                                                      sf=(opt['upscale'] if opt['upscale'] else 1))

            # numpy data
            output_data = np.clip(output_data, 0., 1.)

            # print("target_data, output_data,:", target_data.shape, output_data.shape, target_data.max(), output_data.max(), target_data.dtype, output_data.dtype)
            psnr = batch_PSNR(target_data, output_data, True).item()
            ssim = batch_SSIM(target_data, output_data).item()

            ### save image
            img_name = f"data{idx}_noise{n}_metric@{np.round(psnr, 2)}_{np.round(ssim, 2)}_{model_name}.jpg"
            save_one_image(output_data, save_dir=noise_dir, img_name=img_name)

            # output_to_image(target_data, output_data, input_data, save_img = True, save_index = idx)

            psnr_list.append(psnr)
            ssim_list.append(ssim)
            cuda.empty_cache()
            run_time = time.time() - start_time

            avg_psnr = sum(psnr_list) / len(psnr_list)
            avg_ssim = sum(ssim_list) / len(ssim_list)
            # logger.info(f"Average time {run_time}: PSNR {avg_psnr}, SSIM {avg_ssim}")
            # print('{:->4d}--> {:>5f}s | PSNR:{:<4.2f}dB | SSIM:{:<4.2f}'.format(idx, run_time, psnr, ssim))
            # if idx > 20:
            #     break

        logger.info('<Test Dataset:{} {}, Average PSNR : {:<.2f}dB | Average SSIM : {:<.4f} for test: {}'.format(
            dataset, noise_types, avg_psnr, avg_ssim, opt['pretrain_dir']))
    # del logger


def input_single_image(val_data, net, local_window_dim, sf=1):
    net_input = val_data["L1"][:, :3]  # ([1, 4, 3, 500])
    target = val_data["H"][:, :3].cuda()
    onechennel = False
    if net_input.shape[1] < 3:
        onechennel = True
        net_input = net_input.repeat(1, 3, 1, 1)
        target = target.repeat(1, 3, 1, 1)

    _, _, h_old, w_old = net_input.size()
    h_original = h_old * sf
    w_original = w_old * sf

    multiplier = max(h_old // local_window_dim + 1, w_old // local_window_dim + 1)
    h_pad = (multiplier) * local_window_dim - h_old
    w_pad = (multiplier) * local_window_dim - w_old
    net_input = torch.cat([net_input, torch.flip(net_input, [2])], 2)[:, :, :h_old + h_pad, :]
    net_input = torch.cat([net_input, torch.flip(net_input, [3])], 3)[:, :, :, :w_old + w_pad]

    # pad again if h/w or w/h ratio is bigger than 2
    if h_pad > h_old or w_pad > w_old:
        _, _, h_old, w_old = net_input.size()
        multiplier = max(h_old // local_window_dim + 1, w_old // local_window_dim + 1)
        h_pad = (multiplier) * local_window_dim - h_old
        w_pad = (multiplier) * local_window_dim - w_old
        net_input = torch.cat([net_input, torch.flip(net_input, [2])], 2)[:, :, :h_old + h_pad, :]
        net_input = torch.cat([net_input, torch.flip(net_input, [3])], 3)[:, :, :, :w_old + w_pad]

    ### evaluate
    _, _, new_h, new_w = net_input.size()
    assert new_h == new_w, "Input image should have square dimension"

    net_input = net_input.cuda()

    net.eval()

    with torch.no_grad():
        net_output = test_by_tile(net_input, net, local_window_dim, sf=sf)

    ### crop the output
    net_output = net_output[:, :, :h_original, :w_original]
    output_data = net_output.cpu().detach().numpy()  # B C H W
    output_data = np.transpose(output_data, (0, 2, 3, 1))  # B H W C
    target_data = target.cpu().detach().numpy()
    target_data = np.transpose(target_data, (0, 2, 3, 1))
    input_data = val_data["H"].cpu().detach().numpy()
    input_data = np.transpose(input_data, (0, 2, 3, 1))

    if onechennel:
        # output_data: (1, 832, 992, 3) 1.6719341 torch.Size([1, 3, 1000, 1000]) tensor(65280., device='cuda:0')

        # print("output_data:", output_data.shape, output_data.max(), net_input.shape, net_input.max())
        # output_data = np.mean(output_data, axis=3, keepdims= True) # torch.mean(output_data, dim=1, keepdim=True)
        output_data = np.clip(output_data, 0, 1)

    return target_data, output_data, input_data


def test_by_tile(img, net, window_size, tile=1024, sf=1):
    b, c, h, w = img.size()

    tile = min(tile, h, w)  # None,
    assert tile % window_size == 0, "tile size should be a multiple of window_size"
    tile_overlap = 0

    stride = tile - tile_overlap
    h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
    w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
    E = torch.zeros(b, c, h * sf, w * sf).type_as(img)
    W = torch.zeros_like(E)

    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            in_patch = img[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
            # print("in_patch",in_patch.shape)

            out_patch = net(in_patch)
            out_patch_mask = torch.ones_like(out_patch)

            E[..., h_idx * sf:(h_idx + tile) * sf, w_idx * sf:(w_idx + tile) * sf].add_(out_patch)
            W[..., h_idx * sf:(h_idx + tile) * sf, w_idx * sf:(w_idx + tile) * sf].add_(out_patch_mask)
    output = E.div_(W)

    return output


