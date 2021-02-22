# Modified from one of the examples of the spotipy module: https://github.com/plamere/spotipy/blob/master/examples/artist_recommendations.py
# Documentation for spotipy: https://spotipy.readthedocs.io/en/2.16.1/#

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


client_credentials_manager = SpotifyClientCredentials(client_id='dc13e99fdbeb4ed2a65f61a206e7ddd3',
														client_secret='d24b65ae6ac4493fa5bffdcf8c601142')
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


class ArtistRecommendations():
    """
    Arguments: 
        artist: a string of artist name
        genre: a list of string of one or more genre(s)
        max_tracks: an int of the max number of songs to return
    
    Return:
        a list of dicts of detail info of each recommended track

    """
    def __init__(self, name, genre=None, max_tracks=20):
        self.name = name
        self.genre = genre
        self.max_tracks = max_tracks

    def show_recommendations(self):
        # https://spotipy.readthedocs.io/en/2.16.1/#spotipy.client.Spotify.search
        search_results = sp.search(q='artist:' + self.name, type='artist')
        items = search_results['artists']['items']
        if len(items) == 0:
            print("Can't find that artist", self.artist)
        else:
            artist = items[0]
            # https://spotipy.readthedocs.io/en/2.16.1/#spotipy.client.Spotify.recommendations
            recom_results = sp.recommendations(seed_artists=[artist['id']], seed_genres=self.genre, limit=self.max_tracks)
            return recom_results['tracks']