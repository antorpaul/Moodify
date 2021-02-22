# A basic outline of the front end for the recommendation system, which 
# retrived Top k recommended tracks from Spotify given an artist name and/or genres, 
# get audio features for these tracks from Spotify API, and 
# get lyrics from Genius API (using LyricsGenius client from https://github.com/johnwmillr/LyricsGenius) 
# After that the tracks info are ready to be input to the sentiment analysis system

import lyricsgenius
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from artist_recommendations import ArtistRecommendations

# Spotify API Authentication
client_credentials_manager = SpotifyClientCredentials(client_id='dc13e99fdbeb4ed2a65f61a206e7ddd3',
														client_secret='d24b65ae6ac4493fa5bffdcf8c601142')
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Genius API Authentication from lyricsgenius
genius = lyricsgenius.Genius('wq2Ucg5jGPKuYxzBpk8TBeZmgP6Xav6uEzjGKRYb-18iltoEulO-CFg2NkkSsp2i')


# list to store track info including name, artist, audio features, and lyrics for 
# all tracks from Spotify recommendation system
recom_tracks = []

# Get recommendation tracks from Spotify API
ar = ArtistRecommendations(name="Charlie Puth", genre=['pop'], max_tracks=5)
recom_results = ar.show_recommendations()

# Get audio features from Spotify API, and lyrics from Genius API for all tracks
for track in recom_results:
	# each track info is stored in a dictionary
	track_info = dict()

	track_info['name'] = track['name']
	track_info['artist'] = track['artists'][0]['name']

	# retrive audio features and store
	track_info['audio_features'] = sp.audio_features(track['uri'])

	# retrive lyrics and store
	song = genius.search_song(track_info['artist'], track_info['name'])
	track_info['lyrics'] = song.lyrics

	recom_tracks.append(track_info)

print(recom_tracks)