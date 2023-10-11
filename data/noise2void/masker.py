import numpy as np




def generate_mask(input, ratio=0.9, size_window=(5, 5)):


    size_data = input.shape   #  [1, 3, 48, 48]
    # print("size_data:", size_data)

    num_sample = int(size_data[2] * size_data[3] * (1 - ratio))

    mask = np.ones(size_data)
    negmask = np.zeros(size_data)
    output = input

    # Channel = 3
    # for ich in range(size_data[2]):
    idy_msk = np.random.randint(0, size_data[2], num_sample)
    idx_msk = np.random.randint(0, size_data[3], num_sample)

    idy_neigh = np.random.randint(-size_window[0] // 2 + size_window[0] % 2, size_window[0] // 2 + size_window[0] % 2, num_sample)
    idx_neigh = np.random.randint(-size_window[1] // 2 + size_window[1] % 2, size_window[1] // 2 + size_window[1] % 2, num_sample)

    idy_msk_neigh = idy_msk + idy_neigh
    idx_msk_neigh = idx_msk + idx_neigh

    idy_msk_neigh = idy_msk_neigh + (idy_msk_neigh < 0) * size_data[2] - (idy_msk_neigh >= size_data[2]) * size_data[2]
    idx_msk_neigh = idx_msk_neigh + (idx_msk_neigh < 0) * size_data[3] - (idx_msk_neigh >= size_data[3]) * size_data[3]

    ich = [0, 1, 2]
    for b in range(size_data[0]):
        for i in ich:
            id_msk = (b, i, idy_msk, idx_msk)
            id_msk_neigh = (b, i, idy_msk_neigh, idx_msk_neigh)

            output[id_msk] = input[id_msk_neigh]
            mask[id_msk] = 0.0
            negmask[id_msk] = 1.0

    return output, mask, negmask