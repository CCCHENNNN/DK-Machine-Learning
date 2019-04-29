import pandas as pd
import numpy as np

# Load the train set and the test set
train_df = pd.read_csv('all/train-set.csv')
test_df = pd.read_csv('all/test-set.csv')

train_df.info()
test_df.info()

train_df.head(10)
test_df.head(10)

# Change some attributes of the set in order to have a bette result in the classifier
def change_feature(data_input):
    data = data_input
    data['Ele_minus_VDH'] = data.Elevation - data.Vertical_Distance_To_Hydrology
    data['Ele_plus_VDH'] = data.Elevation + data.Vertical_Distance_To_Hydrology
    # Add some relation between Elevation and Vertical_Distance_To_Hydrology
    data['Distanse_to_Hydrolody'] = (data['Horizontal_Distance_To_Hydrology'] ** 2 + data['Vertical_Distance_To_Hydrology'] ** 2) ** 0.5
    # Calculate the relative distance 
    data['Hydro_plus_Fire'] = data['Horizontal_Distance_To_Hydrology'] + data['Horizontal_Distance_To_Fire_Points']
    data['Hydro_minus_Fire'] = data['Horizontal_Distance_To_Hydrology'] - data['Horizontal_Distance_To_Fire_Points']
    data['Hydro_plus_Road'] = data['Horizontal_Distance_To_Hydrology'] + data['Horizontal_Distance_To_Roadways']
    data['Hydro_minus_Road'] = data['Horizontal_Distance_To_Hydrology'] - data['Horizontal_Distance_To_Roadways']
    data['Fire_plus_Road'] = data['Horizontal_Distance_To_Fire_Points'] + data['Horizontal_Distance_To_Roadways']
    data['Fire_minus_Road'] = data['Horizontal_Distance_To_Fire_Points'] - data['Horizontal_Distance_To_Roadways']
    # Some relations of hydrology, fire point and road

    # Because there a 40 attributes for the soil and 4 attributes for the wilderness area and those values are binary
    # In fact they have only attribute valued 1 and the others are 0 so if we delete them and create a new attribute with int type, the data set will be more precise
    # Get the type number of soil and create it into the new attribute "Soil".
    # Do same thing as "Wilderness_Area"
    data['Soil'] = 0
    for i in range(1, 41):
        data['Soil'] = data['Soil'] + i * data['Soil_Type' + str(i)]
    data['Wilderness_Area'] = 0
    for i in range(1, 5):
        data['Wilderness_Area'] = data['Wilderness_Area'] + i * data['Wilderness_Area' + str(i)]
    for i in range(1, 41):
        data = data.drop(['Soil_Type' + str(i)], axis=1)
    for i in range(1, 5):
        data = data.drop(['Wilderness_Area' + str(i)], axis=1)
    return data

# Get the features for training
def get_features():
    return ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
            'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
            'Horizontal_Distance_To_Fire_Points',
            'Ele_minus_VDH', 'Ele_plus_VDH', 'Distanse_to_Hydrolody', 'Hydro_plus_Fire', 'Hydro_minus_Fire',
            'Hydro_plus_Road',
            'Hydro_minus_Road', 'Fire_plus_Road', 'Fire_minus_Road', 'Soil', 'Wilderness_Area']

train_df = change_feature(train_df)
test_df = change_feature(test_df)

train_df.info()
test_df.info()

features = get_features()

# Get train set's X and Y
x_train = train_df[:][features].values
y_train = train_df['Cover_Type'].values

# Get test set's id and X
test_id = test_df['Id']
x_test = test_df[:][features].values

# Use the ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesClassifier
clf = ExtraTreesClassifier(max_features=0.3, n_estimators=500)

# Training
clf.fit(x_train, y_train)

# Predicting
print("Begin")
output = clf.predict(x_test)
print("Over")

# Get the result and save it into a csv file
result = np.c_[test_id.astype(int), output.astype(int)]
df_result = pd.DataFrame(result[:,0:2], columns=['Id', 'Cover_Type'])
df_result.to_csv('all/forest.csv', index=False)




