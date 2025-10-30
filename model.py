import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, BatchNormalization,
    Reshape, Dense, GRU, Lambda, add
)
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from data_utils import IMG_SHAPE, NUM_CHARS

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = K.permute_dimensions(y_pred, (1, 0, 2))
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def build_model(input_shape=IMG_SHAPE, num_chars=NUM_CHARS):
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')
    x = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal', name='conv1')(input_data)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), name='pool1')(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal', name='conv2')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), name='pool2')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal', name='conv3')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal', name='conv4')(x)
    x = MaxPooling2D(pool_size=(2, 1), name='pool3')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal', name='conv5')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal', name='conv6')(x)
    x = MaxPooling2D(pool_size=(2, 1), name='pool4')(x)
    x = Conv2D(512, (2, 2), padding='valid', activation='relu', kernel_initializer='he_normal', name='conv7')(x)
    x = BatchNormalization()(x)
    squeezed = Lambda(lambda x: K.squeeze(x, 1))(x)
    gru_1 = GRU(256, return_sequences=True, kernel_initializer='he_normal', name='gru1')(squeezed)
    gru_1b = GRU(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(squeezed)
    gru1_merged = add([gru_1, gru_1b])
    gru_2 = GRU(256, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(gru1_merged)
    gru2_merged = add([gru_2, gru_2b])
    outputs = Dense(num_chars, activation='softmax', name='softmax_output')(gru2_merged)
    prediction_model = Model(inputs=input_data, outputs=outputs)
    labels = Input(name='the_labels', shape=[None], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([outputs, labels, input_length, label_length])
    training_model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
    print("Model built successfully.")
    return training_model, prediction_model
