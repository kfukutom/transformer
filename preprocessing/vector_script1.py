import pathlib
import random
import string
import re
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization

# text file authorization
text_file = keras.utils.get_file(
    fname = "jpn_eng.zip",
    origin = "http://storage.googleapis.com/download.tensorflow.org/data/jpn_eng.zip",
    extract = True
)
text_file = pathlib.Path(text_file).parent / "jpn_eng" / "jpn.txt"