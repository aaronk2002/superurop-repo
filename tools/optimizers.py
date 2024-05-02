import torch


class SMDParameters:
    """
    The Stochastic Mirror Descent optimizer class for specific parameters
    """

    @torch.no_grad()
    def __init__(self, params, lr, p):
        """
        Input:
        - params: the list of parameters to optimize
        - lr    : learning rate
        - p     : the p in l_p-SMD
        """
        self.params = params
        self.lr = lr
        self.p = p
        if p <= 1:
            raise Exception("Unable to process potential for p <= 1")

    @torch.no_grad()
    def step(self):
        """
        Perform one step of optimization according to SMD optimization
        rule
        """
        for weight in self.params:
            weight_prime = (
                torch.sign(weight) * torch.abs(weight) ** (self.p - 1.0)
                - self.lr * weight.grad
            )
            weight.copy_(
                torch.sign(weight_prime)
                * torch.abs(weight_prime) ** (1 / (self.p - 1.0))
            )
