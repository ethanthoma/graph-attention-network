from typing import Callable
from tinygrad import Tensor, nn, GlobalCounters
from tinygrad.helpers import getenv, colored
from tqdm import trange

from . import layer
from . import dataset


config = {
    "epochs": 10_000,
    "batch_size": 60,
    "lr": 5e-3,
}


class Model:
    def __init__(self, in_features: int = 1433, n_hidden: int = 8,
                 num_classes: int = 7, alpha: float = 0.2):

        self.layer_one: Callable[[Tensor, Tensor], Tensor] = layer.GATLayer(
            in_features, n_hidden, 8, alpha)
        self.layer_two: Callable[[Tensor, Tensor], Tensor] = layer.GATLayer(
            n_hidden, num_classes, 1, alpha)

    def __call__(self, x: Tensor, adjacency_matrix: Tensor):
        x = self.layer_one(x, adjacency_matrix)
        x = Tensor.elu(x)
        x = self.layer_two(x, adjacency_matrix)
        x = Tensor.softmax(x)
        return x


def train_cora():
    X_train, Y_train, X_test, Y_test = dataset.fetch_cora(True)
    h, adj = X_train

    model = Model()
    opt = nn.optim.Adam(nn.state.get_parameters(model), lr=config["lr"])

    def train_step() -> Tensor:
        with Tensor.train():
            opt.zero_grad()
            samples = Tensor.randint(config["batch_size"], high=h.shape[0])
            loss = model(
                h[samples], adj[samples]).sparse_categorical_crossentropy(
                Y_train[samples]).backward()
            opt.step()
            return loss

    def get_test_acc(
    ) -> Tensor: return (model(X_test[0], X_test[1]).argmax(axis=1) == Y_test).mean()*100

    test_acc = float('nan')

    for i in (t := trange(config["epochs"])):
        GlobalCounters.reset()
        loss = train_step()
        if i % 10 == 9:
            test_acc = get_test_acc().item()
        t.set_description(
            f"loss: {loss.item():6.2f} test_accuracy: {test_acc:5.2f}%")

    if target := getenv("TARGET_EVAL_ACC_PCT", 0.0):
        if test_acc >= target and test_acc != 100.0:
            print(colored(f"{test_acc=} >= {target}", "green"))
        else:
            raise ValueError(colored(f"{test_acc=} < {target}", "red"))
