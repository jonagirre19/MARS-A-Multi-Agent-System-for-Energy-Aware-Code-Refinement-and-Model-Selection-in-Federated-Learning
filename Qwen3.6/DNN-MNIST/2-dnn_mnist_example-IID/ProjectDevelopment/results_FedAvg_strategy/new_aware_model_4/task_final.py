import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr.common.typing import UserConfig
from torch.utils.data import DataLoader, Dataset
import numpy as np


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class Net(nn.Module):
    def __init__(self, input_dim=1 * 28 * 28, mid_dim=100, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, mid_dim)
        self.fc = nn.Linear(mid_dim, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = Swish()(self.fc1(x))
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set (agnostic)."""
    if isinstance(device, str):
        device = torch.device(device)
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()

    running_loss = 0.0
    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
        for _ in range(epochs):
            for data, target in trainloader:  # Desempaquetado posicional
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                output = net(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

    # Calculamos la pérdida promedio por lote y por época
    avg_trainloss = running_loss / (len(trainloader) * epochs)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on any dataset (agnostic)."""
    if isinstance(device, str):
        device = torch.device(device)
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0

    net.eval()  # <--- IMPORTANTE: Añade esto para desactivar Dropout/Batchnorm
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            for data, target in testloader:
                # Aseguramos que los datos sean tensores y estén en el dispositivo correcto
                data, target = data.to(device), target.to(device)

                outputs = net(data)

                # Si la salida es (Batch, 1) y las etiquetas son (Batch), eliminamos dimensión extra
                if outputs.shape == target.shape + (1,):
                    outputs = outputs.squeeze(-1)

                loss += criterion(outputs, target).item()

                # Predicción agnóstica a la forma de la salida
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == target).sum().item()

    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


class NPZDataset(Dataset):
    def __init__(self, path: Path):
        with np.load(path, allow_pickle=True) as loader:
            content = loader["data"].item()

            x_data = content["x"]
            y_data = content["y"]

            self.x = torch.from_numpy(np.array(x_data, dtype=np.float32))
            self.y = torch.from_numpy(np.array(y_data, dtype=np.int64))

            if self.x.ndim == 3:
                self.x = self.x.unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def create_run_dir(config: UserConfig) -> tuple[Path, str]:
    """Create a directory where to save results from this run."""
    # Create output directory given current timestamp
    current_time = datetime.now()
    run_dir = current_time.strftime("%Y-%m-%d/%H-%M-%S")
    # Save path is based on the current directory
    save_path = Path.cwd() / f"outputs/{run_dir}"
    save_path.mkdir(parents=True, exist_ok=False)

    # Save run config as json
    with open(f"{save_path}/run_config.json", "w", encoding="utf-8") as fp:
        json.dump(config, fp)

    return save_path, run_dir


def load_data(partition_id: int, num_partitions: int):
    """Carga datos desde archivos .npz locales de forma genérica."""

    # Define la ruta base donde están tus carpetas 'train' y 'test'
    # Por ejemplo, si están en la raíz de tu proyecto:
    base_path = Path.cwd() / "dnn_mnist_example"
    train_file = base_path / "train" / f"{partition_id}.npz"
    test_file = base_path / "test" / f"{partition_id}.npz"

    # Cargamos los datasets sin transformaciones de imagen
    train_ds = NPZDataset(train_file)
    test_ds = NPZDataset(test_file)

    # Creamos los DataLoaders
    trainloader = DataLoader(train_ds, batch_size=32, shuffle=True)
    testloader = DataLoader(test_ds, batch_size=32)

    return trainloader, testloader
