import pandas as pd
import numpy as np
import tensorflow as tf
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import os

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
NUM_CHARS = len(LETTERS) + 1
IMG_WIDTH = 128
IMG_HEIGHT = 32
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 1)

def load_and_split_data(csv_path, data_dir, test_size=0.1, random_state=42):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    df['Label'] = df['Label'].astype(str)
    max_label_len = df['Label'].str.len().max()
    print(f"Max label length found: {max_label_len}")
    df['Image_Path'] = df['Image_Path'].apply(lambda x: os.path.join(data_dir, x))
    print(f"Splitting data into train and validation sets (test_size={test_size})...")
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")
    return train_df, val_df, max_label_len

def preprocess_image(img_path, img_size=(IMG_WIDTH, IMG_HEIGHT)):
    img = imread(img_path, as_gray=True)
    img = resize(img, (img_size[1], img_size[0]), anti_aliasing=True)
    img = img.T
    img = np.expand_dims(img, axis=-1)
    return img

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size, data_dir, img_shape, max_label_len, 
                 letters=LETTERS, downsample_factor=4):
        self.df = df
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.img_shape = img_shape
        self.max_label_len = max_label_len
        self.letters = letters
        self.letters_map = {letter: i for i, letter in enumerate(self.letters)}
        self.downsample_factor = downsample_factor
        self.cnn_output_len = self.img_shape[1] // self.downsample_factor
        self.indices = self.df.index.tolist()
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_indices]
        X = np.zeros((self.batch_size, self.img_shape[0], self.img_shape[1], 1), dtype=np.float32)
        y = np.full((self.batch_size, self.max_label_len), -1, dtype=np.float32)
        input_length = np.ones((self.batch_size, 1), dtype=np.int64) * self.cnn_output_len
        label_length = np.zeros((self.batch_size, 1), dtype=np.int64)

        for i, row in enumerate(batch_df.itertuples()):
            img_path = row.Image_Path
            img = preprocess_image(img_path, (self.img_shape[1], self.img_shape[0]))
            X[i] = img
            label = str(row.Label)
            label_length[i] = len(label)
            encoded_label = self._encode_to_labels(label)
            y[i, 0:len(encoded_label)] = encoded_label

        inputs = {
            'the_input': X,
            'the_labels': y,
            'input_length': input_length,
            'label_length': label_length
        }
        outputs = {'ctc': np.zeros([self.batch_size])}
        return inputs, outputs

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

    def _encode_to_labels(self, label_str):
        encoded = []
        for char in label_str:
            if char in self.letters_map:
                encoded.append(self.letters_map[char])
        return encoded
