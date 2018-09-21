import logging
from tqdm import tqdm

log = logging.getLogger(__name__)


class Trainer(object):

    def __init__(self, train_loader, num_epochs: int, num_support_tr: int, val_loader=None):
        self.train_loader = train_loader
        self.num_epochs = num_epochs
        self.val_loader = val_loader
        self.num_support_tr = num_support_tr

    def __call__(self, model, optimizer):
        if self.val_loader is None:
            best_state = None
        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []
        best_acc = 0.
        for epoch in range(self.num_epochs):
            log.info(f"=== Epoch {epoch:04d} ===")
            train_iter = iter(self.train_loader)
            model.train()
            for x, y in tqdm(train_iter):
                optimizer.zero_grad()
                y_hat = model(x)
                loss, acc = model.criterion(y_hat, y, self.num_support_tr)
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
                train_acc.append(acc.item())
            if self.val_loader is None:
                continue
        return best_state, best_acc, train_loss, train_acc, val_loss, val_acc
