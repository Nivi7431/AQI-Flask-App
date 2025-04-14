import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Generate dummy data with 5 features
X = np.random.rand(100, 5) * 100
y = np.random.rand(100) * 500  # Fake AQI values

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the model using joblib
joblib.dump(model, 'randomForestRegressor.pkl')
print("✅ Model trained and saved with 5 features!")
