import pandas as pd

# Đọc dữ liệu từ file CSV
df = pd.read_csv('product_group_data.csv')

# Xử lý các giá trị thiếu
df['GroupManager'] = df['GroupManager'].fillna('Unknown')

# Loại bỏ các hàng có giá trị thiếu (nếu cần)
df = df.dropna()

# Chuyển đổi kiểu dữ liệu ngày tháng
df['CreatedDate'] = pd.to_datetime(df['CreatedDate'], errors='coerce')
df['UpdatedDate'] = pd.to_datetime(df['UpdatedDate'], errors='coerce')

# Chuyển đổi cột 'IsActive' thành kiểu boolean
df['IsActive'] = df['IsActive'].astype(bool)

# Đổi tên cột và cập nhật giá trị
df.rename(columns={
    'ProductGroupID': 'Product_Group_ID',
    'GroupName': 'Group_Name',
    'GroupDescription': 'Group_Description',
    'ParentGroupID': 'Parent_Group_ID',
    'GroupCode': 'Group_Code',
    'Category': 'Category',
    'IsActive': 'Status',  # Thay đổi tên cột từ 'IsActive' thành 'Status'
    'CreatedDate': 'Created_Date',
    'UpdatedDate': 'Updated_Date',
    'GroupManager': 'Group_Manager'
}, inplace=True)

# Chuyển đổi các giá trị boolean trong cột 'Status' thành chuỗi mô tả
df['Status'] = df['Status'].replace({True: 'Hoạt động', False: 'Không hoạt động'})

# Loại bỏ các cột không cần thiết (nếu có)
df = df.drop(columns=['Department'])

# Xóa dữ liệu trùng lặp
df = df.drop_duplicates()

# Lưu dữ liệu đã làm sạch vào file CSV mới
df.to_csv('product_group_data_cleaned.csv', index=False)

print("\nDữ liệu đã được làm sạch và lưu vào file 'product_group_data_cleaned.csv'")
