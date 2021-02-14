## Files
* rnn_demo_medium.py - walking through a medium article detailing tensorflow/keras + LSTMs to classify tweets with an array of different emotions
* song_sentiment_analysis_rnn.py - sentiment analsysis model using the data from data_moodsUPDATED.csv to train. Includes word embedding layers and LSTMs and a model using tensorflow's keras.

## Issues
* Current accuracy seems fairly low

### Tensorflow Install Issues
* I had some issues when trying to run my code with Tensorflow/Keras and here's how I went about it.
    * Here is a Git issue that details my problem: https://github.com/tensorflow/tensorflow/issues/45930
    * https://www.tensorflow.org/install/gpu#install_cuda_with_apt
    * Turns out my issues were just bound to happen because I didn't have an nvidia GPU, but tensorflow seems to use my CPU by default when it detects that. I was just worried because of the nasty error message.