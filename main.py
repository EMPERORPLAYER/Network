import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# ---------- Завдання 1: підготовка даних ----------
print("Task 1")
data = load_diabetes()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("X_train:", X_train.shape, "X_test:", X_test.shape)
print("y_train:", y_train.shape, "y_test:", y_test.shape)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)


# ---------- Завдання 2: модель MLP ----------
print("Task 2")
class MLP(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_model(weight_decay, epochs=50, lr=0.001):
    torch.manual_seed(42)
    model = MLP(X_train_t.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train_t), y_train_t)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"  wd={weight_decay} | epoch {epoch+1}/{epochs} | loss={loss.item():.2f}")

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_t).numpy().flatten()
    return y_pred


# ---------- Завдання 3: регуляризація ----------
print("Task 3")
print("\nБез регуляризації:")
pred_no_reg = train_model(weight_decay=0.0)
print("\nЗ регуляризацією (weight_decay=1e-4):")
pred_reg = train_model(weight_decay=1e-4)

results = pd.DataFrame({
    "Конфігурація": ["Без регуляризації", "З регуляризацією (L2)"],
    "MAE": [mean_absolute_error(y_test, pred_no_reg), mean_absolute_error(y_test, pred_reg)],
    "R²": [r2_score(y_test, pred_no_reg), r2_score(y_test, pred_reg)],
})
print("\n", results.to_string(index=False))

# ---------- Завдання 4: scatter-графік ----------
print("Task 4")
plt.figure(figsize=(7, 6))
plt.scatter(y_test, pred_reg, alpha=0.7, label="Передбачення")
lims = [min(y_test.min(), pred_reg.min()), max(y_test.max(), pred_reg.max())]
plt.plot(lims, lims, "r--", label="y = x")
plt.xlabel("Реальні значення (y_test)")
plt.ylabel("Передбачені значення (y_pred)")
plt.title("Diabetes HealthRisk: реальні vs передбачені значення")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("diabetes_healthrisk_analysis.png", dpi=150)
plt.show()