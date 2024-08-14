import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu từ tệp CSV
file_path = 'website_access_data.csv'
data = pd.read_csv(file_path)

# Chuyển đổi cột 'timestamp' thành định dạng ngày tháng
data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')

# Loại bỏ các giá trị NaT (không thể chuyển đổi)
data = data.dropna(subset=['timestamp'])

# Chuyển đổi cột 'duration' từ giây sang phút
data['duration'] = data['duration'].astype(float) / 60

# Biểu đồ 1: Số lượt truy cập vào từng trang web theo thiết bị
device_page_visits = data.groupby(['device', 'page']).size().unstack().fillna(0)

fig, ax1 = plt.subplots(figsize=(14, 8))
device_page_visits.plot(kind='bar', stacked=True, ax=ax1, colormap='viridis')
ax1.set_title('Number of Visits to Pages by Device')
ax1.set_xlabel('Device')
ax1.set_ylabel('Number of Visits')
ax1.legend(title='Page', bbox_to_anchor=(1.05, 1), loc='upper left')

# Thêm số liệu chi tiết lên biểu đồ 1
for container in ax1.containers:
    ax1.bar_label(container, label_type='center')

# Hiển thị biểu đồ 1
plt.tight_layout()
plt.show()

# Biểu đồ 2: Thời gian truy cập trung bình cho từng thiết bị
average_duration = data.groupby(['device'])['duration'].mean()

fig, ax2 = plt.subplots(figsize=(14, 8))
average_duration.plot(kind='bar', ax=ax2, color='skyblue')
ax2.set_title('Average Duration by Device')
ax2.set_xlabel('Device')
ax2.set_ylabel('Average Duration (minutes)')

# Thêm số liệu chi tiết lên biểu đồ 2
for index, value in enumerate(average_duration):
    ax2.text(index, value, f'{value:.2f}', ha='center', va='bottom')

# Hiển thị biểu đồ 2
plt.tight_layout()
plt.show()
