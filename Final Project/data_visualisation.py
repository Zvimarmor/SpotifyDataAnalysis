import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import os
import re

# ========== SETUP ==========
FIG_DIR = "Final Project/figures/data_figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ========== LOAD DATA ==========
df = pd.read_csv("cleaned_data/total_df.csv")

# ========== CLEAN & RENAME ==========
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

# ========== PLOT 1: Popularity by Genre Over Time (Using Real Years) ==========
# Drop rows without year
df = df.dropna(subset=['year'])
df['year'] = df['year'].astype(int)

# Group by genre and year, calculate average popularity
genre_year_popularity = df.groupby(['track_genre', 'year'])['popularity'].mean().reset_index()

# Keep top N most common genres
top_genres = df['track_genre'].value_counts().head(20).index.tolist()
genre_year_popularity = genre_year_popularity[genre_year_popularity['track_genre'].isin(top_genres)]

# Plot
plt.figure(figsize=(20, 10))
sns.lineplot(data=genre_year_popularity, x='year', y='popularity', hue='track_genre',
             palette='tab10', marker='o', linewidth=2.5)

plt.title("Average Popularity Over Time by Genre", fontsize=16)
plt.xlabel("Release Year")
plt.ylabel("Average Popularity")
plt.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/popularity_over_time_by_genre.png")
plt.close()


# ========== PLOT 2: Word Cloud - Cleaned Words in Track Titles ==========
def clean_title(title):
    # Remove common non-title junk
    title = re.sub(r'\(feat[^\)]*\)', '', title, flags=re.IGNORECASE)  # (feat. ...)
    title = re.sub(r'feat[^\s]*', '', title, flags=re.IGNORECASE)      # feat...
    title = re.sub(r'ft[^\s]*', '', title, flags=re.IGNORECASE)        # ft...
    title = re.sub(r'featuring[^\s]*', '', title, flags=re.IGNORECASE) # featuring...
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

# ========== PLOT 3: Word Cloud - Most Common Artist Name Parts ==========
top_songs = df[df['popularity'] >= 70]
artist_names = top_songs['artists'].astype(str).str.lower()

# Replace separators like ; , &
artist_names = artist_names.str.replace(r"[;&]", ",", regex=True)
artist_names = artist_names.str.replace(r"\s+&\s+", ",", regex=True)

# Split to individual names and count name parts
name_parts = []
for entry in artist_names:
    for artist in entry.split(","):
        artist = artist.strip()
        if artist:
            name_parts.extend(artist.split())

name_counts = Counter(name_parts)
filtered_counts = {name: count for name, count in name_counts.items() if count > 2}  # remove too rare

artist_wordcloud = WordCloud(width=1000, height=600, background_color='white',
                             colormap='magma', max_words=100).generate_from_frequencies(filtered_counts)

plt.figure(figsize=(12, 6))
plt.imshow(artist_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Most Frequent Name Parts in Artists of Popular Songs", fontsize=16)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/wordcloud_artist_names_parts.png")
plt.close()

# ========== PLOT 4: Heatmap of Feature Correlations ==========
numerical_features = ['popularity', 'duration_ms','danceability','energy','key',
                      'loudness','mode','speechiness','acousticness',
                      'instrumentalness','liveness','valence','tempo','time_signature']

corr_matrix = df[numerical_features].corr()

# Sorted correlation
pop_corr_sorted = corr_matrix['popularity'].drop('popularity').sort_values(ascending=False)

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True,
            cbar_kws={'label': 'Correlation Coefficient'})
plt.title("Correlation Matrix of Numerical Features")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/correlation_heatmap_full.png")
plt.close()

# ========== EXTRA: Bar Plot of Correlation with Popularity ==========
plt.figure(figsize=(10, 6))
sns.barplot(x=pop_corr_sorted.values, y=pop_corr_sorted.index, palette='crest')
plt.title("Correlation of Features with Popularity")
plt.xlabel("Pearson Correlation")
plt.ylabel("Feature")
plt.axvline(0, color='black', linestyle='--', linewidth=1)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/correlation_with_popularity_barplot.png")
plt.close()
