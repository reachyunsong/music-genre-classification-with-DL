from pyaudio import PyAudio, paInt16 
import numpy as np 
from datetime import datetime 
import wave
import time 

import os
import numpy as np
import librosa
import librosa.display
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns

import tensorflow as tf
if type(tf.contrib) != type(tf): tf.contrib._warning = None

model = load_model('weights.best.h5')



import pandas as pd
df_label = pd.read_csv("./data/result.csv",index_col=0)

df_test = df_label[int(0.9*len(df_label)):]

import os
import numpy as np
import librosa
import librosa.display
import pandas as pd
import matplotlib.pyplot as plt

AUDIO_DIR = './data/fma_test'
def get_audio_path(audio_dir, track_id):
    """
    Return the path to the mp3 given the directory where the audio is stored
    and the track ID.
    Examples
    --------
    >>> import utils
    >>> AUDIO_DIR = os.environ.get('AUDIO_DIR')
    >>> utils.get_audio_path(AUDIO_DIR, 2)
    '../data/fma_small/000/000002.mp3'
    """
    tid_str = '{:06d}'.format(track_id)
    return os.path.join(audio_dir, tid_str[:3], tid_str + '.mp3')

def create_spectogram(track_id):
    filename = get_audio_path(AUDIO_DIR, track_id)
    print(filename)
    y, sr = librosa.load(filename)
    spect = librosa.feature.melspectrogram(y=y, sr=sr,n_fft=2048, hop_length=1024)
    spect = librosa.power_to_db(spect, ref=np.max)
    return spect.T

def create_array(df):
    genres = []
    X_spect = np.empty((0, 640, 128))
    count = 0
    #Code skips records in case of errors
    for index, row in df.iterrows():
        # print("id:",row['tracks_id'])
        try:
            track_id = int(row['tracks_id'])

            spect = create_spectogram(track_id)

            # Normalize for small shape differences
            spect = spect[:640, :]
            X_spect = np.append(X_spect, [spect], axis=0)
            genres.append(row['genres_all'])
            # if count % 100 == 0:
                # print("Currently processing: ", count)
        except:
            count += 1
            # print("Couldn't process: ", count)
            continue
    y_arr = np.array(genres)
    return X_spect, y_arr

x_test, y_test = create_array(df_test)

x_test_raw = librosa.core.db_to_power(x_test, ref=1.0)

x_test = np.log(x_test_raw)

test = np.load('./data/test.npz')

x_test = test['arr_0']

y_test = test['arr_1']

y_test_one = np.array(pd.get_dummies(y_test))

# perdict labels
y_true = np.argmax(y_test_one, axis=1)
x_test = np.expand_dims(x_test, axis = -1)
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
# target_names = ['2', '4','5','10','12','15','17','18','21','25','26','38','181','182','297','468','1235']
target_names = ['International', 'Jazz','Classical','Pop','Rock',
                'Electronic','Folk','Soundtrack','Hip-Hop','Punk',
                'Post-Rock','Experimental','Techno','House','Chip Music',
                'Dubstep','Instrumental']



# get the result
data = {'pred':y_pred, 'true':y_true} 
df_result = pd.DataFrame(data)

print('Sample prediction(head 20):')
print(df_result.head(20))

print('\nClassification Report:')
print(classification_report(y_true, y_pred, target_names=target_names))


from sklearn.metrics import accuracy_score
print('\nAccuracy:')
print(round(accuracy_score(y_true, y_pred),3))


from sklearn.metrics import confusion_matrix
import seaborn as sns

mat = confusion_matrix(y_true, y_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=target_names,
            yticklabels=target_names)
print('\nConfusion Matrix:')
print('Saved in the folder as "confusion_matrix.jpg"')
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.savefig('confusion_matrix.eps',format= 'eps')
plt.show()

