import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

# Bước 1: Load dữ liệu từ file CSV
file_path = 'D:/python/Pandass/market_trend_data.csv'  # Cập nhật đường dẫn tới file của bạn
data = pd.read_csv(file_path)

# Bước 2: Xử lý dữ liệu
# Chuyển đổi các cột chuỗi (categorical columns) sang số sử dụng LabelEncoder
label_encoders = {}
for column in ['TrendDescription', 'MarketSegment', 'Region', 'CompetitorAnalysis', 'TrendSource', 'ImpactLevel']:
    label_enc = LabelEncoder()
    data[column] = label_enc.fit_transform(data[column])
    label_encoders[column] = label_enc

# Xử lý các giá trị thiếu (missing values)
data.fillna(0, inplace=True)  # Với các giá trị thiếu, ta sẽ thay bằng 0 hoặc giá trị trung bình

# Bước 3: Lựa chọn các đặc trưng (features) và biến mục tiêu (target variable)
features = ['ProductGroupID', 'ImpactLevel', 'MarketSegment', 'Region', 'CompetitorAnalysis', 'TrendSource']
X = data[features]
y = data['GrowthRate']

# Chuẩn hóa (scaling) các đặc trưng
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Bước 4: Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Bước 5: Huấn luyện mô hình
model = LinearRegression()
model.fit(X_train, y_train)

# Bước 6: Dự đoán và đánh giá mô hình
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# In hệ số hồi quy và điểm giao nhau
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")

# Bước 7: Tạo biểu đồ so sánh giá trị thực và giá trị dự đoán
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3, color='red')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.title('GrowthRate Actual Value vs. Predicted Value Comparison Chart')
plt.show()
