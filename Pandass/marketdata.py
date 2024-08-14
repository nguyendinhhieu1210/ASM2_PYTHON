import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = 'market_trend_data.csv'
data = pd.read_csv(file_path)

# Group by ProductGroupID and Region, count the number of occurrences
trend_counts = data.groupby(['ProductGroupID', 'Region']).size().unstack().fillna(0)

# Plot the bar chart
fig, ax1 = plt.subplots(figsize=(14, 8))
trend_counts.plot(kind='bar', stacked=True, ax=ax1, colormap='viridis')
ax1.set_title('Number of Trends by ProductGroupID and Region')
ax1.set_xlabel('ProductGroupID')
ax1.set_ylabel('Number of Trends')
ax1.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')

# Add detailed numbers on the chart
for container in ax1.containers:
    ax1.bar_label(container, label_type='center')

# Show the plot
plt.tight_layout()
plt.show()
