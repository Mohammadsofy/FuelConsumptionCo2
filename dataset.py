
#* import the packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
#* read the data
df=pd.read_csv(r"C:\Users\dell\Desktop\python\dala\FuelConsumptionCo2.csv")
#* show the data
print(df.head())
#* show the into of data
print(df.info())
#* Description of Our Data
print(df.describe())
#* duplicated
print(df.duplicated().sum())

#* visualization in histogram to data [CYLINDERS, ENGINESIZE, FUELCONSUMPTION_COMB, CO2EMISSIONS]
visualize_data=df[['CYLINDERS','ENGINESIZE','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
visualize_data.hist()
plt.show()
#* visualization in scatter plot to data [FUELCONSUMPTION_COMB, CO2EMISSIONS]
plt.scatter(x=df["FUELCONSUMPTION_COMB"],y=df["CO2EMISSIONS"])
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("CO2EMISSIONS")
plt.show()
#* visualization in scatter plot to data [ENGINESIZE, CO2EMISSIONS]
plt.scatter(df["ENGINESIZE"], df["CO2EMISSIONS"])
plt.xlabel("ENGINESIZE")
plt.ylabel("CO2EMISSIONS")
plt.show()
# #* visualization in scatter plot to data [CYLINDERS, CO2EMISSIONS]
plt.scatter(df["CYLINDERS"],df["CO2EMISSIONS"])
plt.xlabel("CYLINDERS")
plt.ylabel("CO2EMISSIONS")
plt.show()

msk=np.random.rand(len(df))<0.8
train=df[msk]
test=df[msk]
#* visualization in scatter plot to train of data [ENGINESIZE, CO2EMISSIONS]
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS)
plt.xlabel("ENGINESIZE")
plt.ylabel("CO2EMISSIONS")
plt.show()
#* Create a Linear Regression object
regr = linear_model.LinearRegression()
#* Extract the 'ENGINESIZE' and 'CO2EMISSIONS' columns from the training data
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
#* Train the Linear Regression model on the training data
regr.fit(train_x, train_y)
#* Print the coefficients and intercept of the trained model
print("Coefficients:", regr.coef_)
print("Intercept:", regr.intercept_)
#* Create a scatter plot of the training data
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
#* Plot the regression line using the trained model's coefficients and intercept
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], "-r")
#* Add labels to the plot
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()
#* Extract the 'ENGINESIZE' column from the test data
test_x = np.asanyarray(test[['ENGINESIZE']])
#* Extract the 'CO2EMISSIONS' column from the test data
test_y = np.asanyarray(test[['CO2EMISSIONS']])
#* Use the trained model to make predictions on the test data
test_y_ = regr.predict(test_x)
#* Calculate and print the Mean Absolute Error (MAE)
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
#* Calculate and print the Residual Sum of Squares (MSE)
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
#* Calculate and print the R2-score
print("R2-score: %.2f" % r2_score(test_y, test_y_))
