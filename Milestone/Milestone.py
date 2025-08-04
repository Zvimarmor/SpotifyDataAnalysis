import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# טוענים את הדאטה
df = pd.read_csv("cleaned_data/total_df.csv")

# בוחרים עמודת פופולריות
popularity = df['popularity_y'] if 'popularity_y' in df.columns else df['popularity_x']

# גרף משולב: היסטוגרמה + KDE + קו נורמלי
plt.figure(figsize=(10, 6))
sns.histplot(popularity, kde=True, stat="density", bins=30, color='skyblue', edgecolor='black', label='Actual distribution')

# קו נורמלי תאורטי
mu, std = popularity.mean(), popularity.std()
xmin, xmax = popularity.min(), popularity.max()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'r--', linewidth=2, label='Normal distribution')

# עיצוב
plt.title("Distribution of Song Popularity")
plt.xlabel("Popularity")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("popularity_distribution.png", dpi=300)
plt.show()
