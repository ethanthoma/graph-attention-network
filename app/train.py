from typing import Callable
from tinygrad import Tensor, nn, GlobalCounters
from tinygrad.helpers import getenv, colored
from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict
from tqdm import trange

from . import layer
from . import dataset


config = {
    "epochs": 1_000,
    "batch_size": 500,
    "lr": 5e-3,
    "traning_size": 500,
    "in_features": 1433,
    "n_hidden": 8,
    "num_classes": 7,
    "alpha": 0.2,
    "dropout": 0.6,
}


class Model:
    def __init__(self, in_features: int, n_hidden: int, num_classes: int,
                 alpha: float, dropout: float):

        self.layer_one: Callable[[Tensor, Tensor], Tensor] = layer.GATLayer(
            in_features, n_hidden, 8, alpha, dropout)
        self.layer_two: Callable[[Tensor, Tensor], Tensor] = layer.GATLayer(
            n_hidden, num_classes, 1, alpha, dropout)

    def __call__(self, x: Tensor, adjacency_matrix: Tensor):
        x = self.layer_one(x, adjacency_matrix)
        x = Tensor.elu(x)
        x = self.layer_two(x, adjacency_matrix)
        x = Tensor.softmax(x)
        return x


def train_cora():
    h, adj, labels = dataset.fetch_cora(True)

    model = Model(in_features=config["in_features"],
                  n_hidden=config["n_hidden"],
                  num_classes=config["num_classes"],
                  alpha=config["alpha"],
                  dropout=config["dropout"])
    opt = nn.optim.Adam(nn.state.get_parameters(model), lr=config["lr"])

    def train_step() -> Tensor:
        with Tensor.train():
            opt.zero_grad()
            samples = Tensor.randint(
                config["batch_size"], high=config["traning_size"])

            out = model(h, adj)
            out = out[samples]

            loss = out.sparse_categorical_crossentropy(
                labels[samples]).backward()
            opt.step()
            return loss

    def get_test_acc() -> Tensor:
        test_h = h[config["traning_size"]:]
        test_adj = adj[config["traning_size"]:, config["traning_size"]:]
        test_labels = labels[config["traning_size"]:]
        return (model(test_h, test_adj).argmax(axis=1) == test_labels).mean()*100

    def save_model(model, filename):
        state_dict = get_state_dict(model)
        safe_save(state_dict, filename)

    test_acc = float('nan')

    for i in (t := trange(config["epochs"])):
        GlobalCounters.reset()
        loss = train_step()
        if i % 10 == 9:
            test_acc = get_test_acc().item()
        t.set_description(
            f"loss: {loss.item():6.2f} test_accuracy: {test_acc:5.2f}%")

        if i % 100 == 99:
            save_model(model, f'models/model_epoch_{i+1}.safetensors')

    if target := getenv("TARGET_EVAL_ACC_PCT", 0.0):
        if test_acc >= target and test_acc != 100.0:
            print(colored(f"{test_acc=} >= {target}", "green"))
        else:
            raise ValueError(colored(f"{test_acc=} < {target}", "red"))

    save_model(model, 'models/model.safetensors')
