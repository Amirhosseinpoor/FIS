import os
import gdown
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from itertools import cycle
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# -----------------------------
# Download and Load Data
# -----------------------------
url = "https://drive.google.com/uc?id=15TJ1Kp7SbvaQo6vN5ilKtfbwhI5YhekD"
if not os.path.exists("teleCust1000t.csv"):
    print("Downloading dataset...")
    gdown.download(url, "teleCust1000t.csv", quiet=False)

df = pd.read_csv('teleCust1000t.csv')
print(df.head())
print(df.info())
print(df.describe())

# -----------------------------
# EDA
# -----------------------------
plt.figure(figsize=(10, 10))
sns.heatmap(100 * df.corr(), annot=True)
plt.show()

sns.pairplot(df, vars=['tenure', 'income', 'ed', 'employ'], hue='custcat',
             palette=['blue', 'red', 'green', 'yellow'])
plt.show()

def hexplt(data, x, y, gridsize):
    plt.hexbin(data[x], data[y], gridsize=gridsize, cmap='plasma')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.colorbar()
    plt.show()

hexplt(df, 'tenure', 'income', 10)

plt.figure(figsize=(7, 5))
sns.countplot(data=df, x='custcat')
plt.show()

class_counts = df['custcat'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(class_counts, labels=class_counts.index,
        autopct='%1.1f%%', startangle=90,
        colors=sns.color_palette('viridis', len(class_counts)))
plt.show()

# -----------------------------
# Preprocess
# -----------------------------
X = df.drop('custcat', axis=1)
y = df['custcat'] - 1
X[['region', 'ed', 'reside']] = X[['region', 'ed', 'reside']].astype('category')
new_col = pd.get_dummies(X[['region', 'ed', 'reside']], drop_first=True, dtype=int)
X = pd.concat([X, new_col], axis=1).drop(['region', 'ed', 'reside'], axis=1)
features = X.columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X = np.concatenate((X_train, X_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)

# -----------------------------
# Feature Selection
# -----------------------------
Lasso = LogisticRegression(penalty='l1', solver='saga', max_iter=2000)
Lasso.fit(X, y)
selector1 = SelectFromModel(Lasso, prefit=True)
selected_features1 = features[selector1.get_support()]

Rfe = LogisticRegression()
selector2 = RFE(estimator=Rfe)
selector2.fit(X, y)
selected_features2 = features[selector2.get_support()]

Logistic = LogisticRegression()
Logistic.fit(X_train[:, selector2.get_support()], y_train)

y_hat_train = Logistic.predict(X_train[:, selector2.get_support()])
y_hat_test = Logistic.predict(X_test[:, selector2.get_support()])

print('Accuracy of train:', accuracy_score(y_hat_train, y_train))
print('Accuracy of test:', accuracy_score(y_hat_test, y_test))
print('Confusion Matrix of train:\n', confusion_matrix(y_train, y_hat_train))
print('Confusion Matrix of test:\n', confusion_matrix(y_test, y_hat_test))

# -----------------------------
# ROC Curve (Train & Test)
# -----------------------------
def plot_roc(X_data, y_data, title):
    y_scores = Logistic.predict_proba(X_data)
    classes = np.unique(y_data)
    y_bin = label_binarize(y_data, classes=classes)
    n_classes = y_bin.shape[1]

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure(figsize=(10, 8))
    colors = cycle(['blue', 'red', 'green', 'orange'])
    plt.plot(fpr["macro"], tpr["macro"],
             label=f'Macro-average ROC (AUC = {roc_auc["macro"]:.2f})',
             color='deeppink', linestyle=':', linewidth=4)
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'Class {i + 1} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_roc(X_train[:, selector2.get_support()], y_train, "ROC Curve - Training Data")
plot_roc(X_test[:, selector2.get_support()], y_test, "ROC Curve - Test Data")

# -----------------------------
# PCA
# -----------------------------
pca = PCA(n_components=2)
Z_train = pca.fit_transform(X_train)
Z_test = pca.transform(X_test)
Z_combined = np.concatenate((Z_train, Z_test))
y_combined = np.concatenate((y_train, y_test))
plt.figure(figsize=(10, 8))
colors = ['blue', 'red', 'green', 'orange']
for i, c in enumerate(np.unique(y_combined)):
    plt.scatter(Z_combined[y_combined == c, 0], Z_combined[y_combined == c, 1], c=colors[i], label=f'Class {c + 1}')
plt.legend()
plt.title('PCA Reduction')
plt.show()

# -----------------------------
# LDA
# -----------------------------
lda = LinearDiscriminantAnalysis(n_components=2)
Z_train = lda.fit_transform(X_train, y_train)
Z_test = lda.transform(X_test)
Z_combined = np.concatenate((Z_train, Z_test))
y_combined = np.concatenate((y_train, y_test))
plt.figure(figsize=(10, 8))
for i, c in enumerate(np.unique(y_combined)):
    plt.scatter(Z_combined[y_combined == c, 0], Z_combined[y_combined == c, 1], c=colors[i], label=f'Class {c + 1}')
plt.legend()
plt.title('LDA Reduction')
plt.show()

# -----------------------------
# MLP Embedding
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values)

train_ds = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

class MLP(nn.Module):
    def __init__(self, input_dim, output_classes=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4_embedding = nn.Linear(32, 2)
        self.out = nn.Linear(2, output_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.relu(self.fc3(x))
        z = self.fc4_embedding(x)
        y = self.out(z)
        return y, z

def train_one_epoch(model, loader, loss_fn, optimizer, epoch):
    model.train()
    for inputs, targets in tqdm(loader, desc=f"Epoch {epoch}"):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs, _ = model(inputs)
        loss = loss_fn(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

model = MLP(X_train.shape[1]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1, 301):
    train_one_epoch(model, train_loader, criterion, optimizer, epoch)

_, Z_train = model(X_train_tensor.to(device))
Z_train = Z_train.cpu().detach().numpy()
_, Z_test = model(X_test_tensor.to(device))
Z_test = Z_test.cpu().detach().numpy()
Z_combined = np.concatenate((Z_train, Z_test))
y_combined = np.concatenate((y_train, y_test))

plt.figure(figsize=(10, 8))
for i, c in enumerate(np.unique(y_combined)):
    plt.scatter(Z_combined[y_combined == c, 0], Z_combined[y_combined == c, 1],
                c=colors[i], label=f'Class {c + 1}', alpha=0.7)
plt.legend()
plt.title('MLP Embedding Visualization')
plt.show()
