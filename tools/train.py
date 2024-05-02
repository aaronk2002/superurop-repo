import torch
import numpy as np
from tqdm import tqdm
from tools.optimizers import SMDParameters
from tools.model import AttWModel, AttKQModel
from tools.att_svm import att_svm_solver, att_svm_solver_nuc

eps = torch.finfo(torch.double).eps


def full_train(
    X, Y, z, v, epochs, lr, p, normalized, parameterization, device, std=0.01
):
    """
    Given the sequence of tokesn X, the cross tokens z, the labels Y,
    train a model with lr as learning rate using l_p MD with epochs training
    loops. normalized is boolean determining if we use normalized MD.
    Returns a dictionary with att-svm solution, loss history, parameter
    history, and correlation history
    """
    n, T, d = X.size()

    # Create model
    model = (
        AttWModel(d, std).double()
        if parameterization == "W"
        else AttKQModel(d).double()
    )
    model = model.to(device)
    model.v.data = v
    params = (
        [model.W.weight]
        if parameterization == "W"
        else [model.K.weight, model.Q.weight]
    )

    # Train
    optimizer = SMDParameters(params, lr, p)
    Ws = np.zeros((epochs, d, d))
    losses = np.zeros((epochs,))
    for it in tqdm(range(epochs)):

        # Zero out gradient
        for param in model.parameters():
            param.grad = None

        # Loss calculation
        out = model(X, z).view(-1)
        loss = torch.log(1 + torch.exp(-Y * out))
        loss = loss.mean()

        # Step Optimizer
        if parameterization == "KQ":
            loss += eps * (model.K.weight.norm() ** 2 + model.Q.weight.norm() ** 2)
        loss.backward()
        if normalized:
            if parameterization == "W":
                model.W.weight.grad /= model.W.weight.grad.norm(p / (p - 1)) + eps
            else:
                model.K.weight.grad /= model.K.weight.grad.norm(p / (p - 1)) + eps
                model.Q.weight.grad /= model.Q.weight.grad.norm(p / (p - 1)) + eps
        optimizer.step()

        # Record data
        W = (
            model.W.weight.detach().T
            if parameterization == "W"
            else model.Q.weight.T.mm(model.K.weight).T.detach()
        )
        W = W.to("cpu").numpy()
        sfx_out = model.sfx_out.detach().max(dim=-1)
        ids = sfx_out[1]
        Ws[it] = W
        losses[it] = loss.item()

    # Att-SVM
    X, z, ids = X.to("cpu").numpy(), z.to("cpu").numpy(), ids.to("cpu").numpy()
    sol_att_svm = (
        att_svm_solver(X, z, ids, p)
        if parameterization == "W"
        else att_svm_solver_nuc(X, z, ids)
    )
    sol_att_svm /= np.linalg.norm(sol_att_svm)
    W_corrs = np.zeros((epochs,))
    for it in range(epochs):
        W = Ws[it] / np.linalg.norm(Ws[it])
        W_corrs[it] = sol_att_svm.reshape(-1).dot(W.reshape(-1))

    return {"att-svm": sol_att_svm, "correlations": W_corrs, "Ws": Ws, "losses": losses}
