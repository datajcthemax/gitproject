import pandas as pd

test_data = pd.read_csv('datas/test.csv')

# Check for missing values
print(test_data.isnull().sum())

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
num_features = ['Exercise_Duration', 'Body_Temperature(F)', 'BPM', 'Height(Feet)', 'Height(Remainder_Inches)', 'Weight(lb)', 'Age']
test_data[num_features] = scaler.fit_transform(test_data[num_features])

# One-hot encoding for gender
test_data = pd.get_dummies(test_data, columns=['Gender'])

# Label encoding for weight_status
test_data['Weight_Status'] = test_data['Weight_Status'].replace({'Normal Weight': 1, 'Overweight': 2, 'Obese': 3})



# Select relevant features
selected_features = ['Exercise_Duration', 'Body_Temperature(F)', 'BPM', 'Height(Feet)', 'Weight(lb)', 'Age', 'Weight_Status', 'Gender_F', 'Gender_M']
test_data = test_data[selected_features]

# Save the preprocessed test data
test_data.to_csv('preprocessed_test.csv', index=False)
