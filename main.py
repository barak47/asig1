# main.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load dataset (you can replace this with a dataset of your choice)
data = pd.read_csv('house_prices.csv')

# Preprocess data (example: removing NaNs, selecting features)
X = data[['area', 'bedrooms', 'bathrooms']]  # Example features
y = data['price']  # Target variable

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model for use in Flask API
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved.")

