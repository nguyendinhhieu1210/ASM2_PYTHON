import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = 'market_trend_data.csv'
data = pd.read_csv(file_path)

# Calculate the average GrowthRate by ProductGroupID
average_growth_rate = data.groupby(['ProductGroupID'])['GrowthRate'].mean()

# Calculate the percentage difference in growth rates
percentage_diff = average_growth_rate.pct_change().fillna(0) * 100

# Plot the bar chart
fig, ax1 = plt.subplots(figsize=(14, 8))
average_growth_rate.plot(kind='bar', ax=ax1, color='skyblue')
ax1.set_title('Average Growth Rate by ProductGroupID')
ax1.set_xlabel('ProductGroupID')
ax1.set_ylabel('Average Growth Rate (%)')

# Add detailed numbers on the chart with percentage symbol
for index, value in enumerate(average_growth_rate):
    ax1.text(index, value, f'{value:.2f}%', ha='center', va='bottom')

# Show the plot
plt.tight_layout()
plt.show()
