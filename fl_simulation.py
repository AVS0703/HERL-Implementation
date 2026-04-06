import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import tenseal as ts
import numpy as np

# SIMPLE MODEL (FAST) - no convolutional layers 
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# DATASET
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)

small_dataset, _ = random_split(train_dataset, [2000, len(train_dataset) - 2000])
# ^ we only take 2000 from the entire collection of 60000 
# just so that the program runs faster 
# assumes IID data 

num_clients = 3
sizes = [len(small_dataset) // num_clients] * num_clients
sizes[-1] += len(small_dataset) - sum(sizes) # if the data set doesnt evenly split, excess dumped into last client

client_datasets = random_split(small_dataset, sizes)

client_loaders = [
    DataLoader(ds, batch_size=64, shuffle=True)
    for ds in client_datasets
]

test_loader = DataLoader(small_dataset, batch_size=256, shuffle=False)


# LOCAL TRAINING
def train_local(model, loader, epochs=1, lr=0.01):
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for data, target in loader:
            optimizer.zero_grad() # clear gradients from previous batch 
            output = model(data) # forward pass : make predictions 
            loss = criterion(output, target) # how wrong were we?
            loss.backward() # backward pass : compute gradients 
            optimizer.step() # update weights in the direction that reduces loss


# HE CONTEXT
def create_context(N, q):
    """
    Create a TenSEAL CKKS context for the given polynomial modulus degree N.
    The q parameter is passed in by the RL agent but the actual coefficient
    modulus bit sizes are set per N to satisfy CKKS security constraints.
    """
    if N == 2048:
        coeff_mod_bit_sizes = [30, 20]
    elif N == 4096:
        coeff_mod_bit_sizes = [40, 20, 20]
    else:
        N = 8192
        coeff_mod_bit_sizes = [40, 40, 20, 20]
    
    # picked these values based on the CKKS community standards
    # coeff_mod_bit_sizes comes from the idea that in ring theory, when q is a product of primes 
    # you can use Chinese Remainder Theorem and split computations across each prime independently and then combine 
    # so this is more efficient 

    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=N,
        coeff_mod_bit_sizes=coeff_mod_bit_sizes
    )
    # context is basically like a settings file for your encryption 

    context.generate_galois_keys() #galois keys authorize and enable vector rotation operations 
    # we have split our weights (tensor) into several chunks
    #if you want to sum those chunks across clients during aggregation, you need rotation operations 
    context.global_scale = 2 ** 20

    return context


# ENCRYPT / DECRYPT

def encrypt_vector(context, tensor, N):
    """
    Encrypt a tensor by normalizing it and splitting into safe-sized chunks.
    Returns the encrypted chunks and the normalization scale (max_val).
    """
    flat = tensor.detach().cpu().numpy().astype(float).flatten() 
    # converts a PyTorch model parameter into a flat 1D numpy array 

    # Normalize to prevent scale overflow during HE operations
    max_val = float(np.max(np.abs(flat))) + 1e-8 #1e-8 avoids zero error 
    flat = flat / max_val

    # CKKS can only encrypt upto N/2. we choose 512 to ne conservative
    chunk_size = 512
    chunks = []

    for i in range(0, len(flat), chunk_size):
        chunk = flat[i:i + chunk_size].tolist()
        enc = ts.ckks_vector(context, chunk)
        chunks.append(enc)

    return chunks, max_val


def decrypt_vector(enc_chunks, shape, max_val):
    #Decrypt a list of encrypted chunks and reconstruct a tensor of the given shape.
    decrypted = []

    for enc in enc_chunks:
        decrypted.extend(enc.decrypt())

    num_elements = int(torch.prod(torch.tensor(list(shape))))
    tensor = torch.tensor(decrypted[:num_elements], dtype=torch.float32)
    return (tensor * max_val).view(shape)


# MAIN FL FUNCTION
def run_fl(N, q):
    """
    Run one round of federated learning with homomorphic encryption.

    Args:
        N (int): Polynomial modulus degree for CKKS (2048, 4096, or 8192).
        q (int): Coefficient modulus proxy — influences local training
                 hyperparameters (epochs, lr) to simulate security/cost tradeoff.

    Returns:
        float: Cross-entropy loss on the test set after one FL round.
    """
    context = create_context(N, q)

    # Higher N / q = more secure but heavier → simulate with more local work
    local_epochs = 1 + int(N / 4096)
    lr = max(0.001, 0.01 - (q / 10000))

    global_model = SimpleNet()
    clients = [SimpleNet() for _ in range(num_clients)]

    # Broadcast global model to all clients
    for client in clients:
        client.load_state_dict(global_model.state_dict())

    # -----------------------------
    # LOCAL TRAINING
    # -----------------------------
    for i, client in enumerate(clients):
        train_local(client, client_loaders[i], epochs=local_epochs, lr=lr)

    # -----------------------------
    # ENCRYPT client parameters
    # -----------------------------
    encrypted_params = []

    for client in clients:
        enc_client = []
        for param in client.parameters():
            enc_chunks, max_val = encrypt_vector(context, param.data, N)
            enc_client.append((enc_chunks, param.shape, max_val))
        encrypted_params.append(enc_client)

    # -----------------------------
    # AGGREGATE (FedAvg over encrypted params)
    # -----------------------------
    avg_params = []

    for layer_idx in range(len(encrypted_params[0])):
        num_chunks = len(encrypted_params[0][layer_idx][0])
        shape = encrypted_params[0][layer_idx][1]
        max_val = encrypted_params[0][layer_idx][2]

        avg_layer = []

        for chunk_idx in range(num_chunks):
            # Start sum with client 0's encrypted chunk
            enc_sum = encrypted_params[0][layer_idx][0][chunk_idx]

            # Accumulate remaining clients
            for client_idx in range(1, num_clients):
                enc_sum += encrypted_params[client_idx][layer_idx][0][chunk_idx]

            # BUG FIX: append enc_sum to avg_layer (was missing in original)
            avg_layer.append(enc_sum)

        avg_params.append((avg_layer, shape, max_val))

    # -----------------------------
    # DECRYPT and update global model
    # -----------------------------
    for param, (enc_chunks, shape, max_val) in zip(global_model.parameters(), avg_params):
        # Divide by num_clients to complete the average (FedAvg)
        param.data = decrypt_vector(enc_chunks, shape, max_val) / num_clients

    # -----------------------------
    # EVALUATE loss on test set
    # -----------------------------
    global_model.eval()
    data, target = next(iter(test_loader))

    with torch.no_grad():
        criterion = nn.CrossEntropyLoss()
        output = global_model(data)
        loss = criterion(output, target).item()

    return loss