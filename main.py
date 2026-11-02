import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Створення штучного наберу даних
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

# print("Data shape: ", X.shape)
# print("Label shape: ", y.shape)

scaler = StandardScaler()

X = scaler.fit_transform(X)
# Потрібно дані перетворити в тензор для нейроної мережі
X_tensor = torch.tensor(X, dtype=torch.float)
y_tensor = torch.tensor(y, dtype=torch.float).view(-1, 1)

# print(X_tensor.shape, y_tensor.shape)
#поділ даних на тестові і для тренування
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

#Побудова моделі для нейронної мережі
class MLPClassifier(nn.Module):
    def __init__(self):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(10,32), #z=Wx+b
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,1),
            nn.Sigmoid() #Використовуємо бінарну класифікацію
        )

    # Визначаємо фукнцію для того в якому порядку і як виконувати шари нейронної мережі
    def forward(self, x):
        return self.model(x)

model = MLPClassifier()

#оптимізаціто і фукнцію втрат
criterion = nn.BCELoss() # Бінарна корос етропія
optimizer = optim.SGD(model.parameters()) # Оптиізатор градієнтного спуска

epochs = 200 # За допомогою кількості епох можна порівнюємо результати навчання нашої моделі

for epoch in range(epochs):

    # переводимо модель у режим тренування
    model.train()

    #Обнуляємо попередній градієнт
    optimizer.zero_grad()

    # forward propagation
    outputs = model(X_train)

    #підраховуємо помилку (forward pass)
    loss = criterion(outputs, y_train)

    #підрахунок градієнтів
    loss.backward()

    #оновляємо параметри моделі
    optimizer.step()

    # на кожному етапі показуємо як проводиться навчання нейронної мережі
    with torch.no_grad():
        # переводимо модель у режим інференсу
        model.eval()
        # отримуємо прогнози
        val_outputs = model(X_test)
        val_loss = criterion(val_outputs, y_test)
        accuracy = ((val_outputs > 0.5) == y_test).float().mean()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Accuracy: {accuracy.item():.4f}")