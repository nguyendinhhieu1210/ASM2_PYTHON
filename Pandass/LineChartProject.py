import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu từ tệp CSV
file_path = 'sale_data.csv'
data = pd.read_csv(file_path)

# Chuyển đổi cột 'SaleDate' thành định dạng ngày tháng
data['SaleDate'] = pd.to_datetime(data['SaleDate'])

# Tính tổng số tiền bán hàng cho từng tháng
data['Month'] = data['SaleDate'].dt.to_period('M')
monthly_sales = data.groupby('Month')['TotalAmount'].sum().reset_index()

# Chuyển đổi cột 'Month' về định dạng datetime để vẽ biểu đồ
monthly_sales['Month'] = monthly_sales['Month'].dt.to_timestamp()

# Vẽ biểu đồ cột hiển thị tổng số tiền bán hàng cho từng tháng
plt.figure(figsize=(14, 7))
plt.bar(monthly_sales['Month'], monthly_sales['TotalAmount'], color='skyblue', width=20)

# Vẽ đường thể hiện độ chênh lệch giữa các tháng
plt.plot(monthly_sales['Month'], monthly_sales['TotalAmount'], color='blue', marker='o')

# Thêm tiêu đề và nhãn cho biểu đồ
plt.title('Total Sales Amount per Month')
plt.xlabel('Month')
plt.ylabel('Total Amount')
plt.xticks(rotation=45)
plt.grid(True)

# Hiển thị số liệu trên đỉnh các cột
for i in range(len(monthly_sales)):
    plt.text(monthly_sales['Month'][i], monthly_sales['TotalAmount'][i] + 50,
             f"{monthly_sales['TotalAmount'][i]:.0f}", ha='center', va='bottom')

# Hiển thị biểu đồ
plt.show()
