import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import os
import re

# ===== SETUP =====
FIG_DIR = "Final_project/figures/data_figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ===== LOAD & CLEAN DATA =====
df = pd.read_csv("cleaned_data/total_df.csv")
df = df.drop_duplicates(subset="track_id").dropna()

df = df.rename(columns={
    'popularity_x': 'popularity',
    'duration_ms_x': 'duration_ms',
    'danceability_x': 'danceability',
    'energy_x': 'energy',
    'key_x': 'key',
    'loudness_x': 'loudness',
    'mode_x': 'mode',
    'speechiness_x': 'speechiness',
    'acousticness_x': 'acousticness',
    'instrumentalness_x': 'instrumentalness',
    'liveness_x': 'liveness',
    'valence_x': 'valence',
    'tempo_x': 'tempo',
    'time_signature_x': 'time_signature'
})

columns_to_drop = [
    'track_name_y', 'popularity_y', 'danceability_y', 'energy_y', 'key_y',
    'loudness_y', 'mode_y', 'speechiness_y', 'acousticness_y', 'instrumentalness_y',
    'liveness_y', 'valence_y', 'tempo_y', 'duration_ms_y', 'time_signature_y',
    'Unnamed: 0'
]
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
df = df.dropna(subset=['year'])
df['year'] = df['year'].astype(int)

# ========== PLOT 1: Detect Most Dynamic Genres by Popularity Over Time ==========

# === Parameters ===
NUM_GENRES = 9

# === Compute average popularity per year per genre ===
genre_year_pop = df.groupby(['track_genre', 'year'])['popularity'].mean().reset_index()

# === Compute standard deviation of popularity per genre to detect volatility ===
genre_std = genre_year_pop.groupby('track_genre')['popularity'].std().sort_values(ascending=False)
top_dynamic_genres = genre_std.head(NUM_GENRES).index.tolist()

# === Filter dataset to those genres ===
filtered = genre_year_pop[genre_year_pop['track_genre'].isin(top_dynamic_genres)]

# === Normalize popularity to first year per genre ===
first_year_pop = filtered.groupby('track_genre')['popularity'].transform('first')
filtered['relative_popularity'] = filtered['popularity'] / first_year_pop

plt.figure(figsize=(14, 8))
sns.set_theme(style="whitegrid")
palette = sns.color_palette("tab10", n_colors=len(top_dynamic_genres))

sns.lineplot(
    data=filtered,
    x='year',
    y='relative_popularity',
    hue='track_genre',
    marker='o',
    linewidth=2.5,
    palette=palette
)

plt.title("Top Genres with Most Popularity Change Over Time", fontsize=18)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Relative Popularity (Compared to First Year)", fontsize=14)
plt.xticks(rotation=45)
plt.legend(title="Genre", bbox_to_anchor=(1.01, 1), loc='upper left')
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/top_dynamic_genres_over_time.png")
plt.close()

# ===== PLOT 2: Word Cloud – Popular Track Titles =====
def clean_title(title):
    title = re.sub(r'\(feat[^\)]*\)', '', title, flags=re.IGNORECASE)
    title = re.sub(r'feat[^\s]*', '', title, flags=re.IGNORECASE)
    title = re.sub(r'ft[^\s]*', '', title, flags=re.IGNORECASE)
    title = re.sub(r'featuring[^\s]*', '', title, flags=re.IGNORECASE)
    return title.strip().lower()

popular_titles = df[df['popularity'] >= 70]['track_name_x'].astype(str).apply(clean_title)
title_words = " ".join(popular_titles.tolist())
title_wordcloud = WordCloud(width=1000, height=600, background_color='white',
                            colormap='plasma', max_words=100).generate(title_words)

plt.figure(figsize=(12, 6))
plt.imshow(title_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Most Frequent Words in Popular Track Titles", fontsize=16)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/wordcloud_track_titles_cleaned.png")
plt.close()

# ===== PLOT 3: Word Cloud – Artist Name Parts =====
top_songs = df[df['popularity'] >= 70]
artist_names = top_songs['artists'].astype(str).str.lower()
artist_names = artist_names.str.replace(r"[;&]", ",", regex=True)
artist_names = artist_names.str.replace(r"\s+&\s+", ",", regex=True)

name_parts = []
for entry in artist_names:
    name_parts.extend([name.strip() if name.strip() else "Unknown" for name in entry.split(",")])

name_counts = Counter(name_parts)
filtered_counts = {name: count for name, count in name_counts.items() if count > 2}

artist_wordcloud = WordCloud(width=1000, height=600, background_color='white',
                             colormap='magma', max_words=100).generate_from_frequencies(filtered_counts)

plt.figure(figsize=(12, 6))
plt.imshow(artist_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Most Frequent Name Parts in Artists of Popular Songs", fontsize=16)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/wordcloud_artist_names_parts.png")
plt.close()

# ===== PLOT 4: Feature Correlation Heatmap =====
numerical_features = ['popularity', 'duration_ms','danceability','energy','key',
                      'loudness','mode','speechiness','acousticness',
                      'instrumentalness','liveness','valence','tempo','time_signature']

corr_matrix = df[numerical_features].corr()
pop_corr_sorted = corr_matrix['popularity'].drop('popularity').sort_values(ascending=False)

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True,
            cbar_kws={'label': 'Correlation Coefficient'})
plt.title("Correlation Matrix of Numerical Features")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/correlation_heatmap_full.png")
plt.close()

# ===== PLOT 5: Feature-Popularity Correlation Bar Plot =====
plt.figure(figsize=(10, 6))
sns.barplot(x=pop_corr_sorted.values, y=pop_corr_sorted.index, palette='crest')
plt.title("Correlation of Features with Popularity")
plt.xlabel("Pearson Correlation")
plt.ylabel("Feature")
plt.axvline(0, color='black', linestyle='--', linewidth=1)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/correlation_with_popularity_barplot.png")
plt.close()
