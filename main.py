import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

torch.manual_seed(42)


print("=== Завдання 1 ===")
t1d = torch.tensor([35.0, 78.0, 175.0])                  # вік, вага, зріст одного клієнта
t2d = torch.tensor([[35.0, 78.0, 175.0],
                    [42.0, 90.0, 180.0],
                    [28.0, 60.0, 165.0]])                # 3 клієнти
t3d = torch.randn(2, 3, 3)                               # 2 групи по 3 клієнти

print("1D:", t1d.shape, t1d)
print("2D:", t2d.shape, "\n", t2d)
print("3D:", t3d.shape, "\n", t3d)

age, weight = t2d[:, 0], t2d[:, 1]
print("Додавання:", age + weight)
print("Поелементне множення:", age * weight)
print("Умовний показник ризику:", age * 0.3 + weight * 0.5)

x = torch.tensor([2.0, 3.0], requires_grad=True)
y = (x * x).sum()
print("requires_grad:", x.requires_grad, "| grad_fn:", y.grad_fn)
y.backward()
print("grad:", x.grad)


X_np, y_np = make_classification(
    n_samples=100, n_features=5, n_informative=3, n_redundant=0,
    random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X_np, y_np, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

n_features = X_train.shape[1]

def accuracy(model, X, y):
    with torch.no_grad():
        pred = (torch.sigmoid(model(X)) > 0.5).float()
        return (pred == y).float().mean().item()


print("\n=== Завдання 2 ===")
torch.manual_seed(42)
model = nn.Linear(n_features, 1)
criterion = nn.MSELoss()
lr = 0.01

print("Початкові ваги:", model.weight.data.flatten().numpy().round(4),
      "bias:", round(model.bias.item(), 4))

losses_manual = []
for epoch in range(20):
    pred = model(X_train)
    loss = criterion(pred, y_train)
    losses_manual.append(loss.item())

    model.zero_grad()
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            p -= lr * p.grad

print("Фінальні ваги:  ", model.weight.data.flatten().numpy().round(4),
      "bias:", round(model.bias.item(), 4))
print(f"Loss: {losses_manual[0]:.4f} -> {losses_manual[-1]:.4f}")
print(f"Accuracy (test): {accuracy(model, X_test, y_test):.2f}")


print("\n=== Завдання 3 ===")
torch.manual_seed(42)
model_sgd = nn.Linear(n_features, 1)
optimizer = optim.SGD(model_sgd.parameters(), lr=0.01)
criterion = nn.MSELoss()

losses_sgd = []
print(f"{'Епоха':<8}{'Loss(SGD)':<12}")
for epoch in range(1, 21):
    pred = model_sgd(X_train)
    loss = criterion(pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses_sgd.append(loss.item())
    print(f"{epoch:<8}{loss.item():<12.4f}")

print(f"Accuracy (test): {accuracy(model_sgd, X_test, y_test):.2f}")


print("\n=== Завдання 4 ===")
plt.figure(figsize=(8, 5))
plt.plot(range(1, 21), losses_sgd, marker="o", label="SGD (lr=0.01)")
plt.plot(range(1, 21), losses_manual, marker="s", linestyle="--", label="Ручне оновлення")
plt.xlabel("Епоха")
plt.ylabel("Loss (MSE)")
plt.title("Залежність втрат від епох")
plt.legend()
plt.grid(True)
plt.savefig("loss_healthrisk.png", dpi=150)
plt.show()

print("Фінальні ваги SGD:", model_sgd.weight.data.flatten().numpy().round(4),
      "bias:", round(model_sgd.bias.item(), 4))