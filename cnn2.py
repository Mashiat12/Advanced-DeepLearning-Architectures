# Mashiat Tabassum Khan

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import RandomizedSearchCV
from scikeras.wrappers import KerasClassifier
if not hasattr(KerasClassifier, "__sklearn_tags__"):
    def __sklearn_tags__(self):
        return {"estimator_type": "classifier"}
    KerasClassifier.__sklearn_tags__ = __sklearn_tags__
# Load and preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize and reshape
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Define the model creation function
def create_model(learning_rate=0.001, filters=32, kernel_size=(3, 3), pool_size=(2, 2)):
    model = models.Sequential([
        layers.Conv2D(filters, kernel_size, activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(pool_size=pool_size),
        layers.Conv2D(filters * 2, kernel_size, activation='relu'),
        layers.MaxPooling2D(pool_size=pool_size),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Wrap the model with SciKeras's KerasClassifier
model = KerasClassifier(model=create_model, verbose=0)

# Define hyperparameter grid
param_dist = {
    'model__learning_rate': [0.0001, 0.001, 0.01],
    'model__filters': [32, 64],
    'model__kernel_size': [(3, 3), (5, 5)],
    'model__pool_size': [(2, 2), (3, 3)],
    'batch_size': [32, 64],
    'epochs': [5, 10]
}

# Set up RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=10,
    cv=3,
    verbose=2,
    n_jobs=-1
)

# Run the random search
random_search.fit(x_train, y_train)

# Print best hyperparameters
best_params = random_search.best_params_
print("Best Hyperparameters:", best_params)

# Evaluate best model on test set
best_model = random_search.best_estimator_
test_acc = best_model.score(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

