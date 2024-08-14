import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('market_trend_data.csv')

region_counts = data['Region'].value_counts()

plt.figure(figsize=(8, 8))

plt.pie(region_counts,
        labels=region_counts.index,
        autopct='%1.1f%%',
        colors=plt.cm.Paired(range(len(region_counts))))

plt.title('Distribution of Regions')

plt.show()
