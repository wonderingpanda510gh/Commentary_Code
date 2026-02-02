import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# set the random seed
torch.manual_seed(42)

def build_mlp(in_out_features, depth, init_type):
    # this is the mpl function, we have the in_out_features means the dimension of each layer, depth means the hidden layers that the nerual network have
    layers = []
    linears = []
    for _ in range(depth):
        layer = nn.Linear(in_out_features, in_out_features, bias=True)
        if init_type == "standard":
            # standard initilization used in the paper: U[-1/sqrt(n), 1/sqrt(n)]
            a = 1.0 / (width ** 0.5)
            nn.init.uniform_(layer.weight, -a, a)
            nn.init.zeros_(layer.bias)
        elif init_type == "xavier":
            nn.init.xavier_uniform_(layer.weight)  # normalized (xavier) initialization
            nn.init.zeros_(layer.bias)

        layers.append(layer)
        layers.append(nn.Tanh())
        linears.append(layer)
    model = nn.Sequential(*layers)
    return model, linears

def grad_norms_per_layer(linears):
    # compute the l2 gradient for each layer
    norms = []
    for layer in linears:
        g = layer.weight.grad
        norms.append(g.norm().detach().cpu())
    return norms

# Experiment setup
in_out_features = 128
depth = 12
batch = 256

x = torch.randn(batch, in_out_features)
y = torch.randn(batch, in_out_features)

results = {}
for init in ["standard", "xavier"]:
    model, linears = build_mlp(width=width, depth=depth, init_type=init)
    model.zero_grad(set_to_none=True)
    out = model(x)
    loss = nn.MSELoss()(out, y)
    loss.backward()
    results[init] = grad_norms_per_layer(linears)

# Plot the compare graph
plt.figure()
layers_idx = list(range(1, depth + 1))
plt.plot(layers_idx, results["standard"], marker="o", label="Standard init")
plt.plot(layers_idx, results["xavier"], marker="o", label="Normalized (Xavier) init")
plt.xlabel("Layer)")
plt.ylabel("Gradient L2 norm")
plt.legend()
plt.tight_layout()

out_path = "grad_standard_vs_xavier.png"
plt.savefig(out_path, dpi=200)

