import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

# Read all data from csv file
data = pd.read_csv("student-mat.csv", sep=";")

# Print the first five data values
# print(data.head())

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# print(data.head())

# Label or value to predict based on test data attributes provided above
predict = "G3"

# Array of training data without the label(G3)  === attributes
X = np.array(data.drop([predict], axis=1))

# Array of output from array of training data === labels to predict
Y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, Y, test_size=0.1
)

# The following batch commentmust be only done after the model has trained once and the pickle file generated in directory
# The comment is to avoid multiple trainings as after the first training we have already got the best test model saved in pickle file
"""
best = 0
# Run the training of the model 30 times until we get the best model to save in pickle file for the future new data tests
# N.B: You can increase the number of training times more than 30. The current is sample.
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, Y, test_size=0.1
    )

    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    model_accuracy = linear.score(x_test, y_test)
    print("Our Model Accuracy=", model_accuracy)
    # Use Pickle Model to save the best accuracy model for future use on other training data, no need to retrain the model.
    # The highest scoring model instance will be auto saved as our best for future tests
    # This will save in our directory a pickle file with the highest accurate model for future use.
    if model_accuracy > best:
        best = model_accuracy
        with open("Highest_Scoring_Model.pickle", "wb") as f:
            pickle.dump(linear, f)"""

# Read from our saved pickle file
pickle_in = open("Highest_Scoring_Model.pickle", "rb")

# Load the best pickle into our linear model
linear = pickle.load(pickle_in)

# Coefficients(m) for each training data(X value) as Y=mX+b
print("Coefficients for each training data(X value): \n")
print(linear.coef_)

# Intercept for our training (Y value) as Y=mX+b
print("Intercept for each test data output (Y value): \n")
print(linear.intercept_)

each_student_performance_prediction = linear.predict(x_test)

# Predict other the performances of other new students' test data that we did not train our model on
for x in range(len(each_student_performance_prediction)):
    print(each_student_performance_prediction[x], x_test[x], y_test[x])

# Use matplotlib to plot students performances using scatter plot not line plot G1 on X axis and G3 on Y axis
p = "G1"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade(G3)")
pyplot.show()
