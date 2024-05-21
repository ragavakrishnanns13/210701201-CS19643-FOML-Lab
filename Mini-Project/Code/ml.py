import pandas as pd
from sklearn.model_selection import train_test_split
  
dataset = pd.read_csv("C:\datasets\cancer.csv")

x= dataset.drop(columns=["diagnosis(1=m, 0=b)"])
y = dataset["diagnosis(1=m, 0=b)"]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2)

import tensorflow as tf
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(256, input_shape=x_train.shape[1:], activation = "sigmoid")) 
model.add(tf.keras.layers.Dense(256, activation = "sigmoid")) 
model.add(tf.keras.layers.Dense(1, activation = "sigmoid"))

model.compile(optimizer = "adam", loss= 'binary_crossentropy', metrics = ['accuracy'])

model.fit(x_train, y_train, epochs = 1000)

model.evaluate(x_test, y_test)

model.save(r"/Users/ragavakrishnanns/Downloads/210701201-FOML/Mini-Project/Code/tumour_model.h5")      

# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix
# import tensorflow as tf

# # Load dataset
# dataset = pd.read_csv("C:/datasets/cancer.csv")

# # Split features and target variable
# x = dataset.drop(columns=["diagnosis(1=m, 0=b)"])
# y = dataset["diagnosis(1=m, 0=b)"]

# # Split data into train and test sets
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# # Define the model
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(256, input_shape=x_train.shape[1:], activation="sigmoid"),
#     tf.keras.layers.Dense(256, activation="sigmoid"),
#     tf.keras.layers.Dense(1, activation="sigmoid")
# ])

# # Compile the model
# model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])

# # Train the model and save the history
# history = model.fit(x_train, y_train, epochs=1000, verbose=0)

# # Evaluate the model on test set
# loss, accuracy = model.evaluate(x_test, y_test)
# print("Test Accuracy:", accuracy)

# # Plot accuracy over epochs

# plt.plot(history.history['accuracy'], label='Train Accuracy')
# plt.axhline(y=accuracy, color='r', linestyle='--', label='Test Accuracy')
# plt.text(0, accuracy, f'Test Accuracy: {accuracy:.2f}', color='r', ha='left', va='bottom')
# plt.title('Model Accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(loc='upper left')
# plt.show()

# # Plot confusion matrix
# # Predict probabilities for test set
# y_pred_prob = model.predict(x_test)

# # Convert probabilities to binary predictions
# y_pred = (y_pred_prob > 0.5).astype(int)
# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()

# # Predict probabilities for test set
# y_pred_prob = model.predict(x_test)

# # Convert probabilities to binary predictions
# y_pred = (y_pred_prob > 0.5).astype(int)