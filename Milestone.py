import pandas as pd
import numpy as np

##### importing data #####
df = pd.read_csv('Spotify_Tracks_Dataset.csv', index_col=2)


##### plot the popularity of the tracks vs loudness #####
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.scatter(df['loudness'], df['popularity'], alpha=0.5)
plt.title('Popularity vs Loudness of Spotify Tracks')
plt.xlabel('Loudness (dB)')
plt.ylabel('Popularity')
plt.grid()
plt.savefig('popularity_vs_loudness.png')
plt.show()

##### plot the popularity of the tracks vs danceability #####
plt.figure(figsize=(10, 6))
plt.scatter(df['danceability'], df['popularity'], alpha=0.5, color='orange')
plt.title('Popularity vs Danceability of Spotify Tracks')
plt.xlabel('Danceability')
plt.ylabel('Popularity')
plt.grid()
plt.savefig('popularity_vs_danceability.png')
plt.show()

##### plot the popularity of the tracks vs energy #####
plt.figure(figsize=(10, 6))
plt.scatter(df['energy'], df['popularity'], alpha=0.5, color='green')
plt.title('Popularity vs Energy of Spotify Tracks')
plt.xlabel('Energy')
plt.ylabel('Popularity')
plt.grid()
plt.savefig('popularity_vs_energy.png')
plt.show()
