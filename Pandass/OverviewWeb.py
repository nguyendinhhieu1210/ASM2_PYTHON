import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu từ tệp CSV
file_path = 'website_access_data.csv'
data = pd.read_csv(file_path)

# Chuyển đổi cột 'duration' từ giây sang giờ
data['duration'] = data['duration'].astype(float) / 3600

# Tạo figure và các axes
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
fig.suptitle('Website Access Data Analysis', x=0.5, y=0.98)

# Biểu đồ 1: Số lượt truy cập vào từng trang web theo thiết bị
device_page_visits = data.groupby(['device', 'page']).size().unstack().fillna(0)
device_page_visits.plot(kind='bar', stacked=True, ax=axes[0, 0], colormap='viridis')
axes[0, 0].set_title('Number of Visits to Pages by Device')
axes[0, 0].set_xlabel('Device')
axes[0, 0].set_ylabel('Number of Visits')
axes[0, 0].legend(title='Page', bbox_to_anchor=(1.05, 1), loc='upper left')

# Thêm số liệu chi tiết lên biểu đồ 1
for container in axes[0, 0].containers:
    axes[0, 0].bar_label(container, label_type='center', fontsize=8)

# Biểu đồ 2: Số lượt truy cập từ các nguồn referrer
referrer_visits = data['referrer'].value_counts()
referrer_visits.plot(kind='bar', ax=axes[0, 1], color='skyblue')
axes[0, 1].set_title('Number of Visits by Referrer')
axes[0, 1].set_xlabel('Referrer')
axes[0, 1].set_ylabel('Number of Visits')

# Thêm số liệu chi tiết lên biểu đồ 2
for index, value in enumerate(referrer_visits):
    axes[0, 1].text(index, value, str(value), ha='center', va='bottom', fontsize=8)
axes[0, 1].set_xticklabels(referrer_visits.index, rotation=45, ha='right')

# Biểu đồ 3: Số lượt truy cập từ các chiến dịch khác nhau
campaign_visits = data['campaign'].value_counts()
campaign_visits.plot(kind='bar', ax=axes[1, 0], color='lightgreen')
axes[1, 0].set_title('Number of Visits by Campaign')
axes[1, 0].set_xlabel('Campaign')
axes[1, 0].set_ylabel('Number of Visits')

# Thêm số liệu chi tiết lên biểu đồ 3
for index, value in enumerate(campaign_visits):
    axes[1, 0].text(index, value, str(value), ha='center', va='bottom', fontsize=8)
axes[1, 0].set_xticklabels(campaign_visits.index, rotation=45, ha='right')

# Biểu đồ 4: Số lượt truy cập từ các hệ điều hành khác nhau
os_visits = data['os'].value_counts()
os_visits.plot(kind='bar', ax=axes[1, 1], color='salmon')
axes[1, 1].set_title('Number of Visits by OS')
axes[1, 1].set_xlabel('Operating System')
axes[1, 1].set_ylabel('Number of Visits')

# Thêm số liệu chi tiết lên biểu đồ 4
for index, value in enumerate(os_visits):
    axes[1, 1].text(index, value, str(value), ha='center', va='bottom', fontsize=8)
axes[1, 1].set_xticklabels(os_visits.index, rotation=45, ha='right')

# Điều chỉnh bố cục để tránh chồng chéo
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.subplots_adjust(hspace=0.6, wspace=0.3)  # Điều chỉnh khoảng cách giữa các hàng và cột

plt.show()
