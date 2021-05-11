import numpy as np
import torch


def train_autoencoder(
    train_dataloader,
    validation_dataloader,
    net,
    optimizer,
    scheduler,
    cost_function,
    epochs,
    checkpoint_period,
    device,
    indicator="",
    batchsize=2,
    save_folder="output/",
):
    """
    Train the model

    Parameters
    ----------
    train_dataloader : torch.utils.data.DataLoader
        Loader of the training dataset.
    validation_dataloader : torch.utils.data.DataLoader
        Loader of the validation dataset.
    net : AutoEncoder_2D
        The model.
    optimizer : torch.optim.Optimizer
        Optimizer algorithm.
    scheduler : torch.optim.lr_scheduler
        scheduler for changing the learning rate
    cost_function : function
        Cost function.
    epochs : int
        Number of epoch to training.
    checkpoint_period : int
        Period of epoch to save the model
    device : string
        'cuda' or 'cpu'
    indicator : str, optional
        Addition to the names of saved files. The default is ''.
    batchsize : int, optional
        The default is 2.
    save_folder : str, optional
        folder to save the model files. The default is 'output/'.

    Returns
    -------
    train_error : list (float)
        Cost function for training set at each epoch.
    validation_error : list (float)
        Cost function for validation set at each epoch.

    """

    train_error = []
    validation_error = []
    best_net = None

    for i in range(epochs):
        train_costs = []
        for sample_batched in train_dataloader:

            cost = 0
            optimizer.zero_grad()

            X = sample_batched["data"].float().to(device)
            Y = sample_batched["target"].float().to(device)
            X_hat = net(X)
            cost = cost_function(Y, X_hat)
            cost.backward()
            optimizer.step()

            train_costs.append(cost.cpu().detach().numpy())

            del X, Y, cost
            if torch.cuda.torch.cuda.is_available():
                torch.cuda.empty_cache()

        scheduler.step()

        eps_val = test_autoencoder(
            validation_dataloader, net, cost_function, device
        )  # test on the validation set
        train_error.append(np.mean(train_costs))
        validation_error.append(eps_val)

        # update the best model
        if i > 1 and validation_error[i] < validation_error[i - 1]:
            best_net = net

        # check conditions for early stopping
        if (
            i > 8
            and np.mean(validation_error[-4:]) > np.mean(validation_error[-8:-4])
            and train_error[-1] < validation_error[-1]
        ):
            print("Saving current autoencoder model to disk")
            break

        print(
            "EPOCH = %d, COST = %.6f, validation_error = %.6f"
            % (i + 1, train_error[-1], validation_error[-1])
        )

        # save the model periodically
        if (i + 1) % checkpoint_period == 0:
            print("Saving current autoencoder model to disk")
            torch.save(net, save_folder + "/autoencoder2d_" + str(i + 1) + indicator)

    if best_net:
        net = best_net

    torch.save(net, save_folder + "/autoencoder2d_best" + indicator)

    return train_error, validation_error


def test_autoencoder(
    test_dataloader, net, cost_function, device, savefile=None, indicator=""
):
    """
    Calculate the cost function for the model on a test set

    Parameters
    ----------
    test_dataloader : torch.utils.data.DataLoader
        Loader for the test dataset.
    net : AutoEncoder_2D
        The model.
    cost_function : function
        Cost function.
    device : string
        'cuda' or 'cpu'.
    savefile : str, optional
        path to the model file if to load from the file. The default is None.
    indicator : str, optional
        Addition to the names of saved files. The default is ''.

    Returns
    -------
    float
        the mean test error.

    """

    if savefile:
        net = torch.load(savefile)
        net.eval()

    test_costs = []

    for sample_batched in test_dataloader:

        X_test = sample_batched["data"].float().to(device)
        Y_test = sample_batched["target"].float().to(device)
        Y_pred = net(X_test)
        test_cost = cost_function(Y_test, Y_pred).cpu().detach().numpy()
        test_costs.append(test_cost)

        del X_test, Y_test, Y_pred
        if torch.cuda.torch.cuda.is_available():
            torch.cuda.empty_cache()

    return np.mean(test_costs)
