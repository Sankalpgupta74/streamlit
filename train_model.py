import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Example dataset (tum apna CSV use kar sakte ho)
df = pd.read_csv("your_dataset.csv")

# 🔥 IMPORTANT: multiple features select karo
X = df[['feature1', 'feature2', 'feature3', 'feature4']]
y = df['target']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained with features:", X.columns)
