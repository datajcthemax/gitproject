# 데이터 읽어오기
import pandas as pd

train_data = pd.read_csv('C:\\workspace\\gitproject\\datas\\train.csv')
test_data = pd.read_csv('C:\\workspace\\gitproject\\datas\\test.csv')

# 전처리
train_data.isna().sum()
train_data.columns

train_data.drop(['ID','Height(Remainder_Inches)'], axis=1, inplace=True)
test_data.drop(['ID','Height(Remainder_Inches)'], axis=1, inplace=True)

train_data = pd.get_dummies(train_data, columns = ['Gender','Weight_Status'])
test_data = pd.get_dummies(test_data, columns = ['Gender','Weight_Status'])


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = train_data[['Exercise_Duration','Body_Temperature(F)','BPM','Height(Feet)','Weight(lb)','Age','Gender_F','Gender_M','Weight_Status_Normal Weight','Weight_Status_Obese','Weight_Status_Overweight']]
X_test = test_data[['Exercise_Duration','Body_Temperature(F)','BPM','Height(Feet)','Weight(lb)','Age','Gender_F','Gender_M','Weight_Status_Normal Weight','Weight_Status_Obese','Weight_Status_Overweight']]
y_train = train_data['Calories_Burned']

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train.scaled = scaler.fit_transform(X_train)
X_test.scaled = scaler.transform(X_test)

X_train_scaled = train_data[['Exercise_Duration','Body_Temperature(F)','BPM','Height(Feet)','Weight(lb)','Age','Gender_F','Gender_M','Weight_Status_Normal Weight','Weight_Status_Obese','Weight_Status_Overweight']]
X_train_scaled['Calories_Burned'] = train_data['Calories_Burned']

print(X_test)
print(X_train_scaled)

X_train_scaled.to_csv('2_train.csv', index=False)
X_test.to_csv('2_test.csv',index=False)