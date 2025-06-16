# Design Document for the Stock Data Analysis Project

## Project Overview

This project consists of three main scripts that work together to fetch, process, and analyze stock data:

1. **data_collection.py**: Retrieves data from Yahoo Finance and saves it into an SQLite database.
2. **data_processing.py**: Cleans and scales the data to prepare it for machine learning.
3. **model_training.py**: Trains and evaluates a machine learning model using the processed data.

## Details of Each Script

### data_collection.py

- **What It Does**: This script pulls stock data for `JPY=X` and `QQQ` from Yahoo Finance and stores it in a database.
- **How It Works**:
  - Uses the `yfinance` library to download the data.
  - Saves the data into an SQLite database called `stock_data.db` with separate tables for each ticker.

### data_processing.py

- **What It Does**: Prepares the stock data for machine learning.
- **How It Works**:
  - Fetches the data from the SQLite database.
  - Filters the relevant data, reshapes it, and fills in missing values.
  - Scales the feature data to ensure it's ready for modeling.

### model_training.py

- **What It Does**: Trains a neural network model to predict stock prices and evaluates its performance.
- **How It Works**:
  - Defines a neural network using TensorFlow/Keras with a couple of hidden layers and dropout for regularization.
  - Compiles and trains the model using the processed data.
  - Evaluates the model by calculating the Mean Squared Error and plots the actual vs. predicted prices.

## Design Choices

- **Data Collection**: I chose to use `yfinance` and SQLite for simplicity and efficiency in handling and storing the data. I also used 24 years of data since 2000 was the oldest data available
- **Data Processing**: The data is cleaned and scaled to improve the performance of the machine learning model.
- **Model Training**: I used a neural network with regularization to prevent overfitting and ensure that the model generalizes well.

## Assumptions and Constraints

- **Assumptions**: The data collected is assumed to be in a good format and cleaned properly before processing.
- **Constraints**: The model's accuracy heavily depends on the quality of the data and may need further adjustments and fine-tuning.
