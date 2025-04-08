from ast import literal_eval
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler
data_path = 'book_cleaned_python.csv'
df = pd.read_csv(data_path)

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