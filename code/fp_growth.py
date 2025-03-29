import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from sklearn.preprocessing import MultiLabelBinarizer

#  Đọc dữ liệu từ file CSV
df = pd.read_csv("processed_books_csv.csv")

df["Genres"] = df["Genres"].astype(str)  # Chuyển tất cả về chuỗi để tránh float NaN
df["Genres"] = df["Genres"].str.replace(r"[\[\]']", "", regex=True)  # Loại bỏ dấu []
df["Genres"] = df["Genres"].str.split(", ")  # Chuyển thành danh sách

#  Xử lý lỗi NaN (nếu có dòng nào bị NaN sau bước trên)
df["Genres"] = df["Genres"].apply(lambda x: x if isinstance(x, list) else [])  # Nếu lỗi, đặt danh sách rỗng

#  Mã hóa Genres bằng MultiLabelBinarizer
mlb = MultiLabelBinarizer()
genres_encoded = mlb.fit_transform(df["Genres"])
genres_df = pd.DataFrame(genres_encoded, columns=mlb.classes_)

#  Gộp dữ liệu vào DataFrame gốc
df_encoded = pd.concat([df, genres_df], axis=1)

#  Chỉ giữ lại các cột thể loại để chạy FP-Growth
df_fp = df_encoded[mlb.classes_]

#  Chạy thuật toán FP-Growth để tìm tập phổ biến
frequent_itemsets = fpgrowth(df_fp, min_support=0.01, use_colnames=True)

#  Sinh luật kết hợp từ tập phổ biến
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# s Hiển thị một số luật kết hợp
print(rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(10))
# tính confidence_mean
confidence_mean = rules["confidence"].mean()
print(f"Confidence_mean: {confidence_mean:.4f}")


def recommend_books_by_genre(selected_genre, df, rules, num_recommendations=5):
    # Lấy các thể loại liên quan từ luật kết hợp
    related_rules = rules[rules["antecedents"].apply(lambda x: selected_genre in x)]
    if related_rules.empty:
        print(f"Không tìm thấy gợi ý cho thể loại '{selected_genre}'.")
        return
    
    # Chọn thể loại có lift cao nhất
    top_related_genres = related_rules.sort_values("lift", ascending=False)["consequents"].values[:num_recommendations]
    
    recommended_books = df[df[selected_genre] == 1]  # Lấy sách theo thể loại gốc
    
    # Lấy sách theo các thể loại liên quan
    for genre_set in top_related_genres:
        for genre in genre_set:
            recommended_books = pd.concat([recommended_books, df[df[genre] == 1]])
    
    # Loại bỏ trùng lặp
    recommended_books["Genres"] = recommended_books["Genres"].astype(str)  
    recommended_books = recommended_books.drop_duplicates().sample(n=min(len(recommended_books), num_recommendations), random_state=42)

    print(f"Gợi ý sách dựa trên thể loại '{selected_genre}':")
    for i, row in recommended_books.iterrows():
        print(f"- {row['Book']} (Tác giả: {row['Author']}, Rating: {row['Avg_Rating']})")

#  Thử nghiệm gợi ý sách từ thể loại
recommend_books_by_genre("Fantasy", df_encoded, rules)