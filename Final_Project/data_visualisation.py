import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import os
import re
import numpy as np

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

# ========== PLOT 1: Top-N Genre Families Bump Chart ==========

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats

# Parameters
TOP_N = 5
MIN_YEARS_IN_TOPN = 6  # Only show genres that appear in top 5 for at least 6 years

# Group related genres
def group_genres(genre):
    pop_related = ['cantopop', 'indie-pop', 'j-pop', 'k-pop', 'mandopop', 'pop-film', 'pop', 'power-pop', 'synth-pop']
    rock_related = ['alt-rock', 'hard-rock', 'j-rock', 'psych-rock', 'punk-rock', 'rock-n-roll', 'rock', 'rockabilly']
    house_related = ['chicago-house', 'deep-house', 'house', 'progressive-house']
    metal_related = ['black-metal', 'death-metal', 'heavy-metal', 'metal', 'metalcore']
    latin_related = ['latin', 'latino']  # Group latin and latino together
    
    if genre in pop_related:
        return 'pop-family'
    elif genre in rock_related:
        return 'rock-family'
    elif genre in house_related:
        return 'house-family'
    elif genre in metal_related:
        return 'metal-family'
    elif genre in latin_related:
        return 'latin-family'
    else:
        return genre

# Apply grouping
df['grouped_genre'] = df['track_genre'].apply(group_genres)

# Aggregate yearly popularity per family
year_genre = df.groupby(['year', 'grouped_genre'])['popularity'].mean().reset_index()

# Compute rank for each year (1 = most popular)
year_genre['rank'] = year_genre.groupby('year')['popularity'].rank(ascending=False, method='first')

# Keep only top-N ranks, set others to NaN (breaks lines)
year_genre.loc[year_genre['rank'] > TOP_N, 'rank'] = np.nan

# Filter families that appear in top-N for at least MIN_YEARS_IN_TOPN years
valid_families = year_genre[year_genre['rank'].notna()].groupby('grouped_genre')['year'].nunique()
families_to_keep = valid_families[valid_families >= MIN_YEARS_IN_TOPN].index.tolist()
year_genre = year_genre[year_genre['grouped_genre'].isin(families_to_keep)]

# Create bump chart
plt.figure(figsize=(14, 8))
ax = plt.gca()

# Use distinct colors for each family - manually select highly contrasting colors
distinct_colors = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange  
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # gray
    '#bcbd22',  # olive
    '#17becf',  # cyan
    '#ffbb78',  # light orange
    '#98df8a',  # light green
    '#ff9896',  # light red
    '#c5b0d5',  # light purple
    '#c49c94'   # light brown
]

# Take only as many colors as we need
colors = distinct_colors[:len(families_to_keep)]
family_colors = dict(zip(families_to_keep, colors))

# Plot lines for each family
for family in families_to_keep:
    family_data = year_genre[year_genre['grouped_genre'] == family].sort_values('year')
    
    # Plot the line - smoother curve without markers
    plt.plot(family_data['year'], family_data['rank'], 
             color=family_colors[family], linewidth=4, 
             alpha=0.7, label=family.replace('-', ' ').title(),
             linestyle='-', marker=None)

# Styling
plt.title(f"Top-{TOP_N} Popular Genres By Year (2000-2022)", fontsize=18, fontweight='bold', pad=20)
plt.xlabel("Year", fontsize=14, fontweight='600')
plt.ylabel("Rank (1 = Most Popular)", fontsize=14, fontweight='600')

# Invert y-axis so rank 1 is at top
plt.gca().invert_yaxis()

# Set y-axis ticks to integers 1 through TOP_N
plt.yticks(range(1, TOP_N + 1))

# Clean grid and spines
plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)

# Add legend as color bar next to the graph
plt.legend(title="Genre Families", 
          title_fontsize=12, 
          fontsize=11,
          bbox_to_anchor=(1.05, 1), 
          loc='upper left',
          frameon=True,
          fancybox=False,
          shadow=False,
          borderaxespad=0)

# Adjust layout to accommodate legend
plt.subplots_adjust(right=0.75)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/topN_genre_bump_chart.png", dpi=300, bbox_inches='tight')
plt.close()

# ========== ALTERNATIVE VERSION: Popularity Scores Instead of Rankings ==========
# Uncomment the code below and comment out the above version to show actual popularity scores
#
# # Aggregate yearly popularity per family
# year_genre = df.groupby(['year', 'grouped_genre'])['popularity'].mean().reset_index()
#
# # Compute rank for each year (1 = most popular) - keep for filtering but plot popularity
# year_genre['rank'] = year_genre.groupby('year')['popularity'].rank(ascending=False, method='first')
#
# # Keep only top-N ranks, but preserve popularity scores
# top_n_data = year_genre[year_genre['rank'] <= TOP_N].copy()
#
# # Filter families that appear in top-N for at least MIN_YEARS_IN_TOPN years
# valid_families = top_n_data.groupby('grouped_genre')['year'].nunique()
# families_to_keep = valid_families[valid_families >= MIN_YEARS_IN_TOPN].index.tolist()
# year_genre = top_n_data[top_n_data['grouped_genre'].isin(families_to_keep)]
#
# # For families not in top-N in certain years, set popularity to NaN (breaks lines)
# all_years = sorted(year_genre['year'].unique())
# for family in families_to_keep:
#     family_mask = year_genre['grouped_genre'] == family
#     for year in all_years:
#         year_family_mask = family_mask & (year_genre['year'] == year)
#         if not year_family_mask.any():
#             # Add NaN entry for missing year to break the line
#             new_row = pd.DataFrame({
#                 'year': [year], 
#                 'grouped_genre': [family], 
#                 'popularity': [np.nan],
#                 'rank': [np.nan]
#             })
#             year_genre = pd.concat([year_genre, new_row], ignore_index=True)
#
# # Create popularity-based chart
# plt.figure(figsize=(14, 8))
# ax = plt.gca()
#
# # Plot lines for each family using popularity scores
# for family in families_to_keep:
#     family_data = year_genre[year_genre['grouped_genre'] == family].sort_values('year')
#     
#     # Plot the line using popularity scores instead of ranks
#     plt.plot(family_data['year'], family_data['popularity'], 
#              color=family_colors[family], linewidth=4, 
#              alpha=0.7, label=family.replace('-', ' ').title(),
#              linestyle='-', marker=None)
#
# # Styling for popularity version
# plt.title(f"Top-{TOP_N} Popular Genres By Year (2000-2022) - Popularity Scores", fontsize=18, fontweight='bold', pad=20)
# plt.xlabel("Year", fontsize=14, fontweight='600')
# plt.ylabel("Average Popularity Score", fontsize=14, fontweight='600')
#
# # Set y-axis limits based on data range (no inversion needed)
# y_min = year_genre['popularity'].min() - 2
# y_max = year_genre['popularity'].max() + 2
# plt.ylim(y_min, y_max)

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
fig, ax = plt.subplots(figsize=(14, 10))  # Much larger figure

# Create color mapping based on positive/negative correlation
colors = ['#d62728' if x < 0 else '#2ca02c' for x in pop_corr_sorted.values]

bars = plt.barh(range(len(pop_corr_sorted)), pop_corr_sorted.values, color=colors, alpha=0.8)

# Professional styling
plt.title("Audio Features Correlation with Popularity", fontsize=20, fontweight='bold', pad=30)
plt.xlabel("Pearson Correlation Coefficient", fontsize=16, fontweight='600')
plt.ylabel("Audio Features", fontsize=16, fontweight='600')

# Clean axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)

# Add zero line
plt.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.8)

# Improve tick labels with better spacing
plt.yticks(range(len(pop_corr_sorted)), 
           [feature.replace('_', ' ').title() for feature in pop_corr_sorted.index],
           fontsize=12)
plt.xticks(fontsize=12)

# Add value labels on bars - all on the right side for consistency
for i, (bar, value) in enumerate(zip(bars, pop_corr_sorted.values)):
    # Place all values to the right of their bars for clean layout
    plt.text(value + 0.015, i, f'{value:.3f}', 
             va='center', ha='left', fontsize=10, fontweight='600')

plt.grid(axis='x', alpha=0.3)

# Set explicit margins to ensure no cutoff
plt.subplots_adjust(left=0.3, right=0.95, top=0.88, bottom=0.12)
plt.savefig(f"{FIG_DIR}/correlation_with_popularity_barplot.png", dpi=300, bbox_inches='tight')
plt.close()

# ===== PLOT 6: Audio Feature Evolution Over Time =====
audio_features = ['danceability', 'energy', 'valence', 'acousticness', 'speechiness', 'liveness']

# Calculate yearly averages for each feature
yearly_features = df.groupby('year')[audio_features].mean().reset_index()

# Professional subplot styling
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Audio Feature Evolution Over Time (2000-2022)', fontsize=20, fontweight='bold', y=0.95)

# Color palette for each feature
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

for i, feature in enumerate(audio_features):
    row = i // 3
    col = i % 3
    ax = axes[row, col]
    
    # Plot with professional styling
    ax.plot(yearly_features['year'], yearly_features[feature], 
            color=colors[i], linewidth=3, marker='o', markersize=4, alpha=0.9)
    
    # Clean styling for each subplot
    ax.set_title(f'{feature.replace("_", " ").title()} Evolution', 
                fontsize=16, fontweight='bold', pad=10)
    ax.set_xlabel('Year', fontsize=12, fontweight='600')
    ax.set_ylabel(f'Average {feature.replace("_", " ").title()}', fontsize=12, fontweight='600')
    
    # Grid and axis styling
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    
    # Tick styling
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    
    # Set y-axis limits with some padding
    y_min, y_max = yearly_features[feature].min(), yearly_features[feature].max()
    y_padding = (y_max - y_min) * 0.1
    ax.set_ylim(y_min - y_padding, y_max + y_padding)

plt.tight_layout()
plt.subplots_adjust(top=0.88, hspace=0.35, wspace=0.3)  # More space for titles
plt.savefig(f"{FIG_DIR}/audio_feature_evolution_over_time.png", dpi=300, bbox_inches='tight')
plt.close()

