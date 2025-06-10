import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load training data
# Expected columns: age, wear_score, wood_type, material_type, price
# Additional categorical features can be included

data = pd.read_csv('datasets/price_data.csv')

# One-hot encode categorical features
categorical_cols = ['wood_type', 'material_type']
data = pd.get_dummies(data, columns=categorical_cols)

X = data.drop('price', axis=1)
y = data['price']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
)
model.fit(X_train, y_train)

preds = model.predict(X_val)
print('MAE:', mean_absolute_error(y_val, preds))

# Save model
model.save_model('models/price_predictor.json')
