import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the data
df = pd.read_csv('sale_data.csv')

# Convert SaleDate to datetime format
df['SaleDate'] = pd.to_datetime(df['SaleDate'])

# Convert SaleDate to month format
df['Month'] = df['SaleDate'].dt.to_period('M')

# Group the data by month and sum the TotalAmount
monthly_sales = df.groupby('Month')['TotalAmount'].sum().reset_index()

# Convert Month to numerical format (e.g., number of months since the first month)
monthly_sales['Months_Since_Start'] = (monthly_sales['Month'] - monthly_sales['Month'].min()).apply(lambda x: x.n)

# Prepare the data for training
X = monthly_sales[['Months_Since_Start']]
y = monthly_sales['TotalAmount']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate and print the mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Plot the actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual Sales')
plt.plot(X_test, y_pred, color='red', label='Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.xlabel('Months since first sale')
plt.ylabel('Total Sales Amount')
plt.legend()
plt.show()

# Predict future sales (e.g., next 12 months)
future_months = np.array([X['Months_Since_Start'].max() + i for i in range(1, 13)]).reshape(-1, 1)
future_sales = model.predict(future_months)

# Create the corresponding time period for the next 12 months
future_dates = pd.period_range(monthly_sales['Month'].max() + 1, periods=12, freq='M')

# Plot the future sales predictions
plt.figure(figsize=(10, 6))
plt.plot(monthly_sales['Month'].astype(str), monthly_sales['TotalAmount'], color='blue', label='Historical Sales')
plt.plot(future_dates.astype(str), future_sales, color='green', linestyle='--', label='Predicted Future Sales')
plt.title('Sales Growth Forecast by Month')
plt.xlabel('Month')
plt.ylabel('Total Sales Amount')
plt.xticks(rotation=45)
plt.legend()
plt.show()
