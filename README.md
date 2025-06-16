# Stock Correlation Program


Welcome to the Stock Correlation Program! This project is all about analyzing stock data to predict future prices. Here’s a quick rundown of what you’ll find in this project and how to get started.

## What’s in This Project

- **data_collection.py**: This script grabs stock data from Yahoo Finance and saves it into a SQLite database.
- **data_processing.py**: This script prepares the data for machine learning by cleaning and scaling it.
- **model_training.py**: This script trains a machine learning model to predict stock prices and evaluates its performance.

## How to Get Started

1. **Set Up Your Environment**
   - Make sure you have Python 3.7 or higher.
   - Install the necessary packages by running:
     ```bash
     pip install yfinance pandas numpy tensorflow scikit-learn matplotlib
     ```

2. **Collect the Data**
   - Run the `data_collection.py` script to download stock data and save it to an SQLite database named `stock_data.db`.
     ```bash
     python data_collection.py
     ```

3. **Process the Data**
   - Run the `data_processing.py` script to prepare the data for machine learning. This will output the cleaned and scaled data.
     ```bash
     python data_processing.py
     ```

4. **Train the Model**
   - Run `model_training.py` to train a machine learning model and see how well it performs. The script will print the Mean Squared Error and show a plot comparing actual vs. predicted prices.
     ```bash
     python model_training.py
     ```

## Included Files

- `data_collection.py`: Script to fetch and store data.
- `data_processing.py`: Script to prepare data for training.
- `model_training.py`: Script to train and evaluate the model.
- `README.md`: This guide.
- `DESIGN.md`: A detailed look at the project’s design and decisions.

## Video

- URL: https://youtu.be/9VdNzOuEZ6I