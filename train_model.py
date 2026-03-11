import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("walksafe_data.csv")

# Label encoding
le_lighting = LabelEncoder()
le_crowd = LabelEncoder()
le_time = LabelEncoder()
le_safety = LabelEncoder()

df["lighting_enc"] = le_lighting.fit_transform(df["lighting"])
df["crowd_density_enc"] = le_crowd.fit_transform(df["crowd_density"])
df["time_of_day_enc"] = le_time.fit_transform(df["time_of_day"])
df["safety_level_enc"] = le_safety.fit_transform(df["safety_level"])

# Features and target
X = df[["lighting_enc","crowd_density_enc","incident_count","time_of_day_enc"]]
y = df["safety_level_enc"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train,y_train)

# Save model
joblib.dump(model,"safety_model.pkl")

# Save encoders
joblib.dump({
    "lighting":le_lighting,
    "crowd":le_crowd,
    "time":le_time,
    "safety":le_safety
},"encoders.pkl")

print("Model trained and saved successfully")