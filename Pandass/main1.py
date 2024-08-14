import pandas as pd
import numpy as np

# Đọc dữ liệu từ file CSV
df = pd.read_csv('customer_data.csv')

# Xem thông tin dữ liệu cơ bản
print(df.info())

# Xem vài dòng dữ liệu đầu tiên
print(df.head())

# Xóa các hàng có giá trị NaN trong các cột quan trọng
df = df.dropna(subset=['CustomerID', 'CustomerName'])

# Xóa các cột không cần thiết (nếu có)
df = df.drop(columns=['AccessCategoryID'], errors='ignore')

# Chuyển đổi định dạng ngày tháng (cần xử lý lỗi)
df['DateOfBirth'] = pd.to_datetime(df['DateOfBirth'], errors='coerce')
df['RegistrationDate'] = pd.to_datetime(df['RegistrationDate'], errors='coerce')

# Đổi tên cột nếu cần
df = df.rename(columns={'ContactInfo': 'Contact_Info', 'PhoneNumber': 'Phone_Number'})

# Xử lý giá trị trùng lặp
df = df.drop_duplicates()

# Xử lý dữ liệu không hợp lệ
df = df[df['LoyaltyPoints'].apply(lambda x: pd.notnull(x) and str(x).isdigit())]  # Giữ lại các giá trị số trong LoyaltyPoints
df['LoyaltyPoints'] = df['LoyaltyPoints'].astype(int)

# Lưu dữ liệu đã làm sạch ra file CSV mới
cleaned_file_path = 'cleaned_customer_data_new.csv'
df.to_csv(cleaned_file_path, index=False)

# Đọc dữ liệu từ file CSV mới
df_cleaned = pd.read_csv(cleaned_file_path)

# Kiểm tra các giá trị trống trong cột PreferredContactMethod
print(df_cleaned['PreferredContactMethod'].isna().sum())  # Đếm số lượng giá trị trống

# Xem các giá trị có trong cột PreferredContactMethod
print(df_cleaned['PreferredContactMethod'].value_counts(dropna=False))

# Điền giá trị ngẫu nhiên cho các giá trị 'Unknown' hoặc NaN
contact_methods = ['Email', 'SMS']
df_cleaned['PreferredContactMethod'] = df_cleaned['PreferredContactMethod'].apply(
    lambda x: np.random.choice(contact_methods) if pd.isna(x) or x == 'Unknown' else x
)

# Xem dữ liệu sau khi xử lý
print(df_cleaned['PreferredContactMethod'].value_counts())

# Lưu dữ liệu đã làm sạch ra file CSV mới
df_cleaned.to_csv(cleaned_file_path, index=False)
