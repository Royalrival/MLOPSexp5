import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
import os
import neptune
from neptune.types import File

run = neptune.init_run(
project="bhavvya.jain/Experiment4",
api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkY2FhYmE3My02YzRlLTRlYjctOGM0NS1iMzA4M2Y5NDA1OWYifQ==",

tags='a_tag_to_help_you'
)

lr = LinearRegression()
mse_values = []
#
#############################################################################
# Load and split data
for _ in range(20):
    rng = np.random.RandomState(_)
    x = 10 * rng.rand(1000).reshape(-1,1)
    y = 2 * x - 5 + rng.randn(1000).reshape(-1,1)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)
# Fitting the model

    lr.fit(X_train, y_train)
    test_mse = mean_squared_error(y_test, lr.predict(X_test))
    mse_values.append(test_mse) # Store the MSE value
    print(f'MSE Result: { test_mse}')
    mean_mse = np.mean(mse_values)
    std_mse = np.std(mse_values)
    min_mse = np.min(mse_values)
    max_mse = np.max(mse_values)
    print(f"Mean MSE: {mean_mse:.4f}")
    print(f"Standard Deviation: {std_mse:.4f}")
    print(f"Min MSE: {min_mse:.4f}")
    print(f"Max MSE: {max_mse:.4f}")

    run['test/mse'].log(test_mse)

joblib.dump(lr, '/any_name.pkl')
run['model'].upload(File.as_pickle('any_name.pkl'))