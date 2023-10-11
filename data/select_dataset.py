from data.data_loader import MedDataset, MedTestDataset, MedNoisyOnlyDataset
from torch.utils.data import DataLoader


# ----------------------------------------
# dataloader
# ----------------------------------------
def define_dataset(opt, duplicate_val=False):
    

    Dataset = MedDataset


    if "single_image" in opt['datasets']:
        if opt['datasets']["single_image"]:
            from data.single_data_loader import MedDataset as Dataset

    train_dataset = Dataset(opt, dataset_type="train")    #/Rain100L-Train/
    val_syn_dataset = Dataset(opt, dataset_type="test")

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=opt["dataloader_batch_size"], shuffle=True, pin_memory=True, drop_last=False)
    val_loader = DataLoader(dataset=val_syn_dataset, batch_size=1, pin_memory=False, drop_last=False)


    return train_loader, val_loader, None


def define_test_dataset(opt, datasets, noise_types):

    if opt['noisyonly']:
        dataset = MedNoisyOnlyDataset(opt, datasets)
    else:
        dataset = MedTestDataset(opt, datasets, noise_types)

    loader = DataLoader(dataset=dataset, batch_size=1, pin_memory=False, drop_last=False)
    
    return loader


