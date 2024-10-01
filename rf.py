from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data import prepare_data

df = prepare_data()

# Define the features (X) and the target variable (y) before scaling
X = df.drop(columns=["next_diff"])  # All columns except "next_diff"
y = df["next_diff"]

# Initialize the scaler and fit_transform only on the feature set (X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Initialize the RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

correct = 0

# Print the ratio that the model predicts the correct trend
for i in range(len(y_pred)):
    if (y_pred[i] > 0 and y_test.values[i] > 0) or (
        y_pred[i] < 0 and y_test.values[i] < 0
    ):
        correct += 1

print(f"Correct percentage: {(correct / len(y_pred)) * 100}%")
