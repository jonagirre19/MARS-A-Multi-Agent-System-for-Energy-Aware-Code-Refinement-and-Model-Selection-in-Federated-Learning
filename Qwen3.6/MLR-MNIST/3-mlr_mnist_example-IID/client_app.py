"""pytorch-example: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from mlr_mnist_example.task import Net, load_data
from mlr_mnist_example.task import test as test_fn
from mlr_mnist_example.task import train as train_fn

# Flower ClientApp
app = ClientApp()
classification_head_name = "classification-head"

MODEL_STATE_NAME = "local-model-state"

def save_model_to_state(state: RecordDict, net: torch.nn.Module):
    """Guarda todo el state_dict del modelo de forma agnóstica."""
    state[MODEL_STATE_NAME] = ArrayRecord(net.state_dict())

def load_model_from_state(state: RecordDict, net: torch.nn.Module):
    """Carga los pesos guardados localmente en el modelo."""
    if MODEL_STATE_NAME not in state:
        return

    local_state_dict = state[MODEL_STATE_NAME].to_torch_state_dict()
    net.load_state_dict(local_state_dict, strict=False)


@app.train()
def train(msg: Message, context: Context):
    """Entrenamiento agnóstico."""
    model = Net()
    
    # 1. Aplicamos pesos globales (lo que manda el servidor)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    
    # 2. Aplicamos persistencia local (opcional, para personalización)
    # load_model_from_state(context.state, model)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 3. Carga de datos
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, _ = load_data(int(partition_id), int(num_partitions))

    # 4. Entrenamiento (la función train_fn ya es agnóstica)
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )

    # 5. Guardar estado para la siguiente ronda
    save_model_to_state(context.state, model)

    # 6. Devolver respuesta
    model_record = ArrayRecord(model.state_dict())
    metrics = MetricRecord({
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    })
    return Message(content=RecordDict({"arrays": model_record, "metrics": metrics}), reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data (agnostic version)."""

    model = Net()
    
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    
    # load_model_from_state(context.state, model)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    
    _, valloader = load_data(partition_id, num_partitions)

    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # 6. Construir y devolver el mensaje de respuesta con métricas
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    
    return Message(
        content=RecordDict({
            "metrics": MetricRecord(metrics)
        }), 
        reply_to=msg
    )
