import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 1: Get data from API
response = requests.get("https://restcountries.com/v3.1/all")
data = response.json()

# Step 2: Extract relevant fields
rows = []
for country in data:
    try:
        name = country['name']['common']
        population = country.get('population', None)
        area = country.get('area', None)
        region = country.get('region', None)

        if population and area and region:
            rows.append({
                'name': name,
                'population': population,
                'area': area,
                'region': region
            })
    except KeyError:
        continue

df = pd.DataFrame(rows)

# Step 3: Convert target to binary: Europe vs non-Europe
df['is_europe'] = df['region'].apply(lambda x: 1 if x == 'Europe' else 0)

# Step 4: Preprocess
X = df[['population', 'area']]
y = df['is_europe']

# Step 5: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 7: Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
