"""pytorch-example: A Flower / PyTorch app."""

import torch
from datasets import load_dataset
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from dnn_mnist_example.strategy import CustomStrategy
from dnn_mnist_example.task import Net, create_run_dir, test
import numpy as np

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    # Read run config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_train = context.run_config["fraction-train"]
    fraction_eval = context.run_config["fraction-evaluate"]
    device = context.run_config["server-device"]

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    strategy = CustomStrategy(
        fraction_train=fraction_train, fraction_evaluate=fraction_eval
    )

    # Define directory for results and save config
    save_path, run_dir = create_run_dir(config=context.run_config)
    strategy.set_save_path_and_run_dir(save_path, run_dir)

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": 0.005}),
        num_rounds=num_rounds,
        evaluate_fn=get_global_evaluate_fn(device=device),
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")

def get_global_evaluate_fn(device: str):
    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        print(f'ESTAMOS EN LA RONDA {server_round}')
        test_dir = Path.cwd() / "dnn_mnist_example" / "test" # Asegúrate que esta ruta es la correcta
        print(f'TEST DIR ES {test_dir}')
        npz_files = list(test_dir.glob("*.npz"))
        print(f'Los npz files son {npz_files}')
        
        all_x, all_y = [], []
        
        for file in npz_files:
            try:
                with np.load(file, allow_pickle=True) as loader:
                    content = loader['data'].item() 
                    
                    x_part = content['x']
                    y_part = content['y']
                    
                    all_x.append(np.array(x_part))
                    all_y.append(np.array(y_part))
            except Exception as e:
                print(f"❌ Error leyendo {file.name}: {e}")

        if not all_x:
            print("🚨 ERROR CRÍTICO: No se encontraron datos válidos (.npz con 2 arrays) en la carpeta de test.")
            return MetricRecord({"accuracy": 0.0, "loss": 0.0})

        # Unimos todo en un solo bloque de tensores
        combined_x = torch.from_numpy(np.concatenate(all_x, axis=0)).float()
        combined_y = torch.from_numpy(np.concatenate(all_y, axis=0)).long()
        
        dataset = TensorDataset(combined_x, combined_y)
        testloader = DataLoader(dataset, batch_size=32)

        # Configuración del modelo
        net = Net()
        net.load_state_dict(arrays.to_torch_state_dict())
        net.to(device)
        
        # Aquí llamas a tu función test agnóstica
        loss, accuracy = test(net, testloader, device=device)
        
        return MetricRecord({"accuracy": accuracy, "loss": loss})

    return global_evaluate