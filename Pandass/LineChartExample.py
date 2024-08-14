import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Thiết lập random seed để tái hiện dữ liệu
np.random.seed(0)

# Số lượng mẫu
n_samples = 100

# Dữ liệu cho nhà đất
square_footage_land = np.random.randint(1500, 4000, n_samples)
num_bedrooms_land = np.random.randint(1, 6, n_samples)
age_of_house_land = np.random.randint(0, 50, n_samples)
noise_land = np.random.normal(0, 50000, n_samples)
house_prices_land = square_footage_land * 600 + num_bedrooms_land * 15000 - age_of_house_land * 300 + noise_land

# Tạo DataFrame cho nhà đất
data_land = pd.DataFrame({
    'SquareFootage': square_footage_land,
    'NumBedrooms': num_bedrooms_land,
    'AgeOfHouse': age_of_house_land,
    'HousePrice': house_prices_land,
    'Type': 'Land House'
})

# Dữ liệu cho chung cư
square_footage_condo = np.random.randint(500, 2000, n_samples)
num_bedrooms_condo = np.random.randint(1, 4, n_samples)
age_of_house_condo = np.random.randint(0, 50, n_samples)
noise_condo = np.random.normal(0, 20000, n_samples)
house_prices_condo = square_footage_condo * 300 + num_bedrooms_condo * 7000 - age_of_house_condo * 150 + noise_condo

# Tạo DataFrame cho chung cư
data_condo = pd.DataFrame({
    'SquareFootage': square_footage_condo,
    'NumBedrooms': num_bedrooms_condo,
    'AgeOfHouse': age_of_house_condo,
    'HousePrice': house_prices_condo,
    'Type': 'Condo'
})

# Kết hợp cả hai dữ liệu
data_combined = pd.concat([data_land, data_condo])

# Chuẩn bị dữ liệu để huấn luyện mô hình
X = data_combined[['SquareFootage', 'NumBedrooms', 'AgeOfHouse']]
y = data_combined['HousePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Chuẩn hóa các đặc trưng
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Huấn luyện mô hình hồi quy tuyến tính
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Dự đoán giá cho các cấu hình phòng khác nhau
room_configurations_land = pd.DataFrame({
    'SquareFootage': [2000, 2500, 3000],
    'NumBedrooms': [1, 2, 3],
    'AgeOfHouse': [10, 10, 10],
    'Type': ['Land House', 'Land House', 'Land House']
})
room_configurations_condo = pd.DataFrame({
    'SquareFootage': [1000, 1200, 1500],
    'NumBedrooms': [1, 2, 3],
    'AgeOfHouse': [10, 10, 10],
    'Type': ['Condo', 'Condo', 'Condo']
})

# Kết hợp các cấu hình
room_configurations = pd.concat([room_configurations_land, room_configurations_condo])

# Chuẩn hóa các đặc trưng để dự đoán
room_configurations_scaled = scaler.transform(room_configurations[['SquareFootage', 'NumBedrooms', 'AgeOfHouse']])
predicted_prices = model.predict(room_configurations_scaled)
room_configurations['PredictedPrice'] = predicted_prices

# Vẽ biểu đồ so sánh giá dự đoán cho nhà đất và chung cư dựa trên số lượng phòng ngủ
plt.figure(figsize=(12, 6))
for house_type in room_configurations['Type'].unique():
    subset = room_configurations[room_configurations['Type'] == house_type]
    plt.plot(subset['NumBedrooms'], subset['PredictedPrice'], marker='o', linestyle='-', label=house_type)

# Thêm tiêu đề và nhãn
plt.title('Predicted House Prices for Land Houses and Condos Based on Number of Bedrooms')
plt.xlabel('Number of Bedrooms')
plt.ylabel('Predicted House Price')
plt.legend()
plt.grid(True)

# Thêm các số chênh lệch giá giữa nhà đất và chung cư
for num_bedrooms in room_configurations['NumBedrooms'].unique():
    price_land = room_configurations[(room_configurations['NumBedrooms'] == num_bedrooms) & (room_configurations['Type'] == 'Land House')]['PredictedPrice'].values[0]
    price_condo = room_configurations[(room_configurations['NumBedrooms'] == num_bedrooms) & (room_configurations['Type'] == 'Condo')]['PredictedPrice'].values[0]
    price_diff = price_land - price_condo
    plt.text(num_bedrooms, (price_land + price_condo) / 2, f'{price_diff:.0f}', ha='center', va='bottom', fontsize=10)

plt.show()
