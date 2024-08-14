import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Load the data from the CSV file
data = pd.read_csv('sale_data.csv')

# Extract the relevant data for the scatter plot
X = data['Quantity'].values.reshape(-1, 1)  # Reshape Quantity to a 2D array for sklearn
y = data['TotalAmount'].values  # TotalAmount as the target

# Create a scatter plot
plt.figure(figsize=(10, 6))  # Set the size of the plot
plt.scatter(X, y, color='blue', edgecolor='k', alpha=0.7, s=80)  # Scatter plot with customized points

# Fit a simple linear regression model to illustrate the trend
model = LinearRegression()  # Initialize the linear regression model
model.fit(X, y)  # Fit the model using the data
y_pred = model.predict(X)  # Predict y values based on the model

# Plot the regression line
plt.plot(X, y_pred, color='red', linewidth=2, label='Linear Regression Fit')  # Regression line

# Customize the plot with titles and labels
plt.title('Scatter Plot of Quantity vs. Total Amount with Linear Regression')
plt.xlabel('Quantity')
plt.ylabel('Total Amount')
plt.legend()  # Add legend to the plot
plt.grid(True)  # Add grid for better readability
plt.tight_layout()  # Adjust layout to make room for labels and titles
plt.show()  # Display the plot
