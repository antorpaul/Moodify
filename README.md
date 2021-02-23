## Files
* song_sentiment_analysis_rnn.py - sentiment analsysis model using the data from data_moodsUPDATED.csv to train. Includes word embedding layers and LSTMs and a model using tensorflow's keras.
    * A lot needs to be changed to work better for our song analysis. The current code as of (8/13) reflects more the basic system for tweets which was about a 32% accuracy. I'm hoping that without the nuances of Tweets/twitter culture, it will be easier to go through lyrics. Modern songs will however be tougher so I'll look to see if there are some newer word embeddings with slang and stuff along that lines.
* venv - virtual environment that I've been running the code on


## Issues
* Current accuracy seems fairly low

### Tensorflow Install Issues
* I had some issues when trying to run my code with Tensorflow/Keras and here's how I went about it.
    * Here is a Git issue that details my problem: https://github.com/tensorflow/tensorflow/issues/45930
    * https://www.tensorflow.org/install/gpu#install_cuda_with_apt
    * Turns out my issues were just bound to happen because I didn't have an nvidia GPU, but tensorflow seems to use my CPU by default when it detects that. I was just worried because of the nasty error message.

### TO-DOs
Feb 23, 2021:
#### Rohit
   * Baseline algorithm based off features
#### Yechu
   * Take a look at the rnn
#### Antor
   * Take a look at transformers and bag of words
\Next meet up: Thursday Morning (25th)
