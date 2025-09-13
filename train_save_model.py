# train_save_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# 1. Load your dataset
# Replace with your actual dataset path
data = pd.read_csv("ckd_dataset.csv")

# 2. Encode categorical variables
categorical_cols = [
    'gender', 'hypertension', 'diabetes_mellitus', 'coronary_artery_disease',
    'anemia', 'pedal_edema', 'pus_cell', 'pus_cell_clumps', 'bacteria', 'appetite'
]

# Encode each categorical column
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

# 3. Split into features & target
X = data.drop("class", axis=1)  # Replace "class" with your target column name
y = data["class"]

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Save the model safely
joblib.dump(model, "best_model.pkl", compress=3, protocol=4)
print("Model trained and saved as best_model.pkl")
