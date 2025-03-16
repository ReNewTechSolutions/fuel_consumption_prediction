import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
df = pd.read_csv(url)

# Select input feature (Engine Size) and target output (CO2 Emissions)
cdf = df[['ENGINESIZE', 'CO2EMISSIONS']]
X = cdf[['ENGINESIZE']].to_numpy()  # 2D array
y = cdf['CO2EMISSIONS'].to_numpy()  # 1D array

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Print the model parameters
print('Coefficient:', regressor.coef_[0])
print('Intercept:', regressor.intercept_)

# Function to plot the regression line
def plot_regression(X_train, y_train, model):
    """
    Plots the regression line along with the scatter plot of training data.

    Parameters:
    X_train (array): Training input feature (Engine Size).
    y_train (array): Training target output (CO2 Emissions).
    model (LinearRegression object): Trained Linear Regression model.
    """
    plt.scatter(X_train, y_train, color='blue', alpha=0.5, label="Training Data")
    plt.plot(X_train, model.predict(X_train), '-r', label="Regression Line")
    plt.xlabel("Engine Size")
    plt.ylabel("CO2 Emissions")
    plt.title("Linear Regression: Engine Size vs CO2 Emissions")
    plt.legend()
    plt.show()

# Call the function to plot the regression line
plot_regression(X_train, y_train, regressor)

