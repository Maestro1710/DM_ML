import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
#Avg_Rating
data = pd.read_csv("goodreads_data (1).csv")
print(data.columns)
sns.displot(data['Avg_Rating'], kde=True)
plt.title("Avg_Rating")
plt.show()
#biểu đồ Num_Ratings 
data = pd.read_csv("goodreads_data (1).csv")
print(data.columns)
sns.displot(data['Num_Ratings'], kde=True)
plt.title("Num_Ratings")
plt.show() 