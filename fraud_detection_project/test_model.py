import joblib
import numpy as np

model = joblib.load("fraud_model.pkl")

sample = np.random.rand(1, model.n_features_in_)

prediction = model.predict(sample)

print("Prediction:", "Fraud" if prediction[0] == 1 else "Safe")