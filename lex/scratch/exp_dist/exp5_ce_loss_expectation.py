import torch
import torch.nn.functional as F
from torch import nn


def recreate_ce_loss():
    """
    Simply test the CrossEntropyLoss and your handcrafted implementation.
    The outputs should match up.

    Given:
    - x: (n_classes,)
    - y: scalar  # The target class index
    - y_hat: (n_classes,) the raw output from the model before softmax
    Cross entropy in pure python/numpy code without functions is:
    e_powers = [  # (n_classes,)
        e ** y_hat_i   # Simply makes the values positive. You could use anything ** y_hat_i
        for y_hat_i in y_hat
    ]
    softmax = e_powers / sum(e_powers)  # (n_classes,)  # Normalize the values to sum to 1 and [0, 1] range
    loss = -log(softmax[y])  # scalar  # inf at 0, positive at (0, 1), 0 at 1, negative at > 1. So high loss if close to 0, low loss if close to 1
    """
    x = torch.randn(5)  # (n_classes,)
    y = torch.randint(0, 5, ())  # (scalar)
    y_hat = torch.randn(5, requires_grad=True)  # (n_classes,)

    # Print the inputs and outputs
    print(f"{x=}\n{y=}\n{y_hat=}")

    # Get the loss
    loss = F.cross_entropy(y_hat, y)
    print(f"{loss=}")

    # Get the loss manually
    e_powers = torch.e ** y_hat  # (n_classes,)  # Make them positive. Alternatively, you can use ReLU or any other function.
    softmax = e_powers / e_powers.sum()  # (n_classes,)  # Normalize to [0-1], sum = 1
    loss_manual = -torch.log(softmax[y])  # scalar  # High at 0, low at 1
    # loss_manual = 1 / softmax[y]  # scalar  # High at 0, low at 1
    print(f"{loss_manual=}")

    assert torch.allclose(loss, loss_manual)

    # For a totally uniform distribution
    softmax = 1 / 5 * torch.ones(5, requires_grad=True)
    loss = F.cross_entropy(softmax, y)
    print(f"{loss=}")

    loss_manual = -torch.log(torch.tensor(1 / 5))
    print(f"{loss_manual=}")
    assert torch.allclose(loss, loss_manual)

    print("Success!")

def test_10_classes_uniform():
    """
    We want to test CrossEntropyLoss.
    For 10 classes, what is the CE loss for a uniform distribution?
    In the beginning, when the model is not trained, the model should predict a uniform distribution.
    So this is like the baseline loss.
    """

def test_10_classes_weighted():
    """
    What if the model only predicts the first class?
    Is the loss gonna be higher than uniform distribution?
    """


if __name__ == '__main__':
    recreate_ce_loss()
    # test_10_classes_uniform()
    # test_10_classes_weighted()
