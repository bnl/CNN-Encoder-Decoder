import numpy as np
import random
import torch
from nets import AutoEncoder_2D

# import torch.nn.functional as F
from torch.utils.data import Dataset


class CorrDataSet(Dataset):

    """To use CorrDataSet:

    ds = CorrDataSet(data_file)
    dataloader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    for sample_batched in dataloader:
        do_something(sample)
    """

    def __init__(self, data_file):
        self.data_dict = torch.load(data_file)
        self.uids = list(self.data_dict.keys())
        np.random.shuffle(self.uids)

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):

        uid = self.uids[idx]
        raw_data_path = self.data_dict[uid]["data"]
        target_path = self.data_dict[uid]["target"]
        raw_data = torch.load(raw_data_path)
        target_data = torch.load(target_path)
        sample = {"data": raw_data, "target": target_data}

        return sample


def one_time(Y):
    """
    Calculates one-time correlation function
    from a two-time correlation function

    Parameters
    ----------
    Y : torch tensor
        a two-time correlation function. Shape is (N_roi, N_frames, N_frames).

    Returns
    -------
    res : torch tensor
        one-time correlation function, excluding delay=0. Shape is (N_roi, N_frames-1).

    """
    bs = Y.shape[0]
    step = Y.shape[-1]
    res = torch.zeros(bs, step - 1)
    for i in range(1, step):
        res[:, i - 1] = (
            torch.diagonal(Y, offset=i, dim1=2, dim2=3).mean(axis=2).view(bs)
        )
    return res


def double_cost(Y_out, Y_t):
    """
    Calculates the cost function, which includes both the MSE(2TCF) and MSE(1TCF)

    Parameters
    ----------
    Y_out : torch tensor
        Model output. Shape is (batch_size, N_frames, N_frames).
    Y_t : torch tensor
        Target. Shape is (batch_size, N_frames, N_frames).

    Returns
    -------
    cost function

    """
    return torch.mean((Y_out - Y_t) ** 2) + torch.mean(
        (one_time(Y_out) - one_time(Y_t)) ** 2
    )


def one_time_cost(Y_out, Y_t):
    """ Returns the MSE of 1TCF between the model output Y_out and the target Y_t. """

    return torch.mean((one_time(Y_out) - one_time(Y_t)) ** 2)


def setup_nn(
    dataloader,
    latent_space_dimn=2,
    lr=0.001,
    savefile=None,
    device="cpu",
    weight_decay=0,
    ch1=10,
    ch2=10,
    k=1,
):
    """
    Initialize the model and all its attributes needed for training.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        loader for the training dataset.
    latent_space_dimn : int, optional
        The default is 2.
    lr : int, optional
        learning rate. The default is 0.001.
    savefile : str , optional
        name of the file to load the model from. The default is None.
    device : str, optional
        'cpu' or 'cuda'. The default is 'cpu'.
    weight_decay : float, optional
        regularization parameter for Adam optimized. The default is 0.
    ch1 : int, optional
        number of channels in the first hidden layer of encoder. The default is 10.
    ch2 : int, optional
        number of channels in the second hidden layer of encoder. The default is 10.
    k : int, optional
        kernel size. The default is 1.

    Returns
    -------
    net : AutoEncoder_2D
        The model.
    optimizer : torch.optim.Optimizer
        Optimizer algorithm.
    scheduler : torch.optim.lr_scheduler
        Scheduler for updating the learning rate.
    cost_function : function
        Cost function.

    """
    if savefile is not None:
        net = torch.load(savefile)
        net.eval()

    else:
        print(f"Running on {device}")
        ksize = [k, k]
        channels = [1, ch1, ch2]
        X = next(iter(dataloader))["data"]
        dim_tensor = list(X.shape)
        net = AutoEncoder_2D(dim_tensor, channels, ksize, latent_space_dimn).to(device)

    cost_function = double_cost
    optimizer = torch.optim.Adam(
        net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9995)

    return net, optimizer, scheduler, cost_function


def set_seed(seed):
    """
    Fixing random seed for all libraries.
    """

    import os

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
