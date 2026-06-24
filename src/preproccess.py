from sklearn.preprocessing import StandardScaler

# 1. Initialize the scaler
scaler = StandardScaler()

# 2. We don't want to scale the 'Outcome' or 'Cluster' if they exist, just the features.
# Make sure we are only scaling the 8 biological columns.
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# 3. Apply the scaling
# We create a new dataframe so we don't destroy our original one
df_scaled = df.copy()
df_scaled[features] = scaler.fit_transform(df[features])

# 4. Verify the change
# You will notice the numbers are now decimals, mostly between -3 and 3
print(df_scaled.head())