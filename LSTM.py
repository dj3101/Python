import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score,confusion_matrix

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense

data = pd.read_csv("Fraud_Data_for_test2_1.csv")
df = pd.DataFrame(data)
df = df.dropna(axis=0)

status_percentage = df['Status'].value_counts() / len(df)
weight_map = {
    'Accepted No Fraud': 1.0 / status_percentage['Accepted No Fraud'],
    'Fraud': 1.0 / status_percentage['Fraud'],
    'Blocked': 1.0 / status_percentage['Blocked']
}

df['Weight'] = df['Status'].map(weight_map)
df['Country'] = df['Country'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4})
df['Currency'] = df['Currency'].map({'AD': 0, 'ML': 1, 'SH': 2})
df['Status'] = df['Status'].map({'Accepted No Fraud': 0, 'Blocked': 1, 'Fraud': 2})

X = df[["Time", "Country", "Value", "Currency", "Weight"]].values
y = df[["Status"]].values

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1]).astype('float32')
X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1]).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1]).astype('float32')

y_train = to_categorical(y_train, num_classes=3).astype('float32')
y_val = to_categorical(y_val, num_classes=3).astype('float32')
y_test = to_categorical(y_test, num_classes=3).astype('float32')

model = Sequential([
    LSTM(units=50,input_shape=(X_train.shape[1],X_train.shape[2])),
    Dense(units=3,activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    epochs=100,
                    batch_size=16,
                    validation_data=(X_val, y_val))

train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
epochs = range(1, len(train_loss) + 1)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracy, 'b', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

loss = model.evaluate(X_test, y_test)
predictions = model.predict(X_test)

auc = roc_auc_score(y_test, predictions, multi_class='ovr')

y_test_classes = np.argmax(y_test, axis=1)
predicted_classes = np.argmax(predictions, axis=1)

precision = precision_score(y_test_classes, predicted_classes, average='weighted')
recall = recall_score(y_test_classes, predicted_classes, average='weighted')
f1 = f1_score(y_test_classes, predicted_classes, average='weighted')

print(f"AUC per class: {auc * 100:.2f}%")
print(f'Precision: {precision * 100:.2f}%')
print(f'Recall: {recall * 100:.2f}%')
print(f'F1-score: {f1 * 100:.2f}%')


cm = confusion_matrix(y_test_classes, predicted_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Accepted No Fraud', 'Blocked', 'Fraud'],
            yticklabels=['Accepted No Fraud', 'Blocked', 'Fraud'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
