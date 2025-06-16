import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from data_processing import fetch_data, process_data

# Defines, compiles, and trains a neural network model.
def train_model(X_train, y_train, epochs=100):
    # Define the model
    model = tf.keras.Sequential([
        # Input layer with shape matching the feature size
        tf.keras.Input(shape=(X_train.shape[1],)),
        # Hidden layer with 128 units and ReLU activation, L2 regularization
        tf.keras.layers.Dense(units=128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        # Dropout layer to prevent overfitting
        tf.keras.layers.Dropout(0.2),
        # Hidden layer with 64 units and ReLU activation, L2 regularization
        tf.keras.layers.Dense(units=64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        # Output layer with 1 unit for regression
        tf.keras.layers.Dense(units=1)
    ])
    
    # Compile the model with Adam optimizer and mean squared error loss
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mean_squared_error')

    # Train the model with training data, using 20% of data for validation
    history = model.fit(X_train, y_train, epochs=epochs, validation_split=0.2, verbose=1)
    
    return model, history

# Evaluates the model on test data and visualizes the predictions
def evaluate_model(model, X_test, y_test):
    # Predict on test data
    predictions = model.predict(X_test)

    # Calculate Mean Squared Error between predictions and true values
    mse = np.mean((predictions.flatten() - y_test)**2)
    print(f"Mean Squared Error: {mse}")

    # Visualize the predictions vs actual
    plt.figure(figsize=(10, 5))
    plt.plot(y_test, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.legend()
    plt.title('Actual vs Predicted Prices')
    plt.xlabel('Test Data Points')
    plt.ylabel('USD/JPY Price')
    plt.show()

if __name__ == "__main__":
    # Fetch and process data
    data = fetch_data()
    X, y = process_data(data)
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    best_mse = float('inf')
    best_model = None
    # Train and evaluate 5 models to find the best one
    for i in range(5):
        print(f"Training run {i+1}")
        model, history = train_model(X_train, y_train)
        evaluate_model(model, X_test, y_test)
        # Calculate and compare MSE to keep track of the best model
        predictions = model.predict(X_test)
        mse = np.mean((predictions.flatten() - y_test)**2)
        if mse < best_mse:
            best_mse = mse
            best_model = model

    print(f"Best Mean Squared Error: {best_mse}")
