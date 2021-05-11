import torch
from utils import setup_nn, set_seed, one_time_cost, CorrDataSet
from train_and_test import train_autoencoder
from torch.utils.data import DataLoader


if __name__ == "__main__":

    # define parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data_file = "logs/dataset_training"
    validation_data_file = "logs/dataset_validation"

    epochs = 60
    checkpoint_period = 60
    savefile = None
    lr = 0.00001
    batchsize = 8
    latent_space_dimn = 20
    ch1 = 10
    ch2 = 10
    k = 1
    wd = 0
    seed = 0

    save_folder = "output/"
    indicator = f"_lr_{lr}_latent_space_{latent_space_dimn}_batchsize_{batchsize}_cv_{ch1}_{ch2}_k_{k}_{k}_wd_{wd}"

    set_seed(seed)

    # load the dataset
    train_ds = CorrDataSet(train_data_file)
    val_ds = CorrDataSet(validation_data_file)

    train_dataloader = DataLoader(
        train_ds, batch_size=batchsize, shuffle=True, num_workers=0
    )
    validation_dataloader = DataLoader(
        val_ds, batch_size=batchsize, shuffle=False, num_workers=0
    )

    # initialize everythig for the training
    model, optimizer, scheduler, cost_function = setup_nn(
        train_dataloader, latent_space_dimn, lr, savefile, device, wd, ch1, ch2, k
    )

    # train the model
    T, V = train_autoencoder(
        train_dataloader,
        validation_dataloader,
        model,
        optimizer,
        scheduler,
        cost_function,
        epochs,
        checkpoint_period,
        device,
        indicator,
        batchsize,
        save_folder,
    )

    torch.save(T, save_folder + "/train_error" + indicator)
    torch.save(V, save_folder + "/validation_error" + indicator)
