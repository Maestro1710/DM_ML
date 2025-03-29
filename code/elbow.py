from ast import literal_eval
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler
data_path = 'goodreads_data (1).csv'
df = pd.read_csv(data_path)

# Loại bỏ các cột không cần thiết
df.drop(columns=['Num_Ratings', 'Description', 'URL'], inplace=True)

# Loại bỏ dữ liệu trống
df.dropna(inplace=True)
# df.to_csv("processed_books_csv.csv", index=False)

# Chuẩn hóa cột Genres (loại bỏ dấu [])
df['Genres'] = df['Genres'].apply(lambda x: ', '.join(literal_eval(x)) if isinstance(x, str) else x)
df.to_csv("processed_books_csv.csv", index=False)

# One-hot encoding cho Genres
genres_dummies = df['Genres'].str.get_dummies(sep=', ')

df = pd.concat([df, genres_dummies], axis=1)
df.drop(columns=['Genres'], inplace=True)

# Chuẩn hóa Avg-Rating
scaler = StandardScaler()
df['Avg_Rating'] = scaler.fit_transform(df[['Avg_Rating']])

# Chuyển đổi Author thành dạng số (mã hóa catcategorical encoding)
df['Author'] = df['Author'].astype('category').cat.codes

# tìm số cụm tối ưu bằng elbow method
inertia = []
k_values = range(2, 11)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df[['Author', 'Avg_Rating'] + list(genres_dummies.columns)])
    inertia.append(kmeans.inertia_)
# vẽ elbow 
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o')
plt.xlabel('số cụm (k)')
plt.ylabel('inertia')
plt.title('elbow method để chọn số cụm tối ưu')
plt.show()