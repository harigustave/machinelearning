import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")
# print(data.head())

# Convert all non-numerical  values(high, low, yes, no etc...) into numbers using preprocessor module
le = preprocessing.LabelEncoder()

# Get a list of all non-numerical values from any column with numerical OR non-numeric values and transform them into numerics
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

# print(buying)

predict = "class"

# Put all training attributes into a single tuple
X = list(zip(buying, maint, door, persons, lug_boot, safety))

# Output /test value
Y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

# print(x_train, y_test)

# n_neighbors is same as value of K which is the number of points to compare distance with the training point
# You can increase it ofr decrease it in order to change the model accuracy
model = KNeighborsClassifier(n_neighbors=9)

model.fit(x_train, y_train)
model_accuracy = model.score(x_test, y_test)
print("Current Model Accuracy: ", model_accuracy)

predicted = model.predict(x_test)
names = ["unaccurate", "accurate", "good", "very good"]

for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Data:", x_test[x], "Actual:", names[y_test[x]])

    # Print the distance for each point
    # n=model.kneighbors([x_test[x]], 9, True)
    # print("N: ", n)


