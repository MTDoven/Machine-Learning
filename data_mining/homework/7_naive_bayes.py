import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

data = {
    "Weather": ["Sunny", "Sunny", "Cloudy", "Rainy", "Rainy", "Rainy", "Cloudy", "Sunny", "Sunny", "Rainy", "Sunny", "Cloudy", "Cloudy", "Rainy"],
    "Temperature": [85, 80, 83, 70, 68, 65, 64, 72, 69, 75, 75, 72, 81, 71],
    "Humidity": [85, 90, 78, 96, 80, 70, 65, 95, 70, 80, 70, 90, 75, 80],
    "Windy": ["No", "Yes", "No", "No", "No", "Yes", "Yes", "No", "No", "No", "Yes", "Yes", "No", "Yes"],
    "Play": ["No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"]
}

df = pd.DataFrame(data)
df_encoded = pd.get_dummies(df, columns=["Weather", "Temperature", "Humidity", "Windy"], drop_first=True)

X = df_encoded.drop("Play", axis=1)
y = df["Play"].map({"No": 0, "Yes": 1})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))

probabilities = model.predict_proba(X_test)
for i, prob in enumerate(probabilities):
    print(f"Sample {i + 1} probability: Yes={prob[0]:.2f}, No={prob[1]:.2f}")
