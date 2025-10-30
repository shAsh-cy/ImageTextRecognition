import argparse
import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from data_utils import load_and_split_data, DataGenerator, IMG_SHAPE, LETTERS, NUM_CHARS
from model import build_model

def main(args):
    print("--- 1. Loading and splitting data ---")
    train_df, val_df, max_label_len = load_and_split_data(args.data_path, args.data_dir)
    print("--- 2. Creating data generators ---")
    downsample_factor = 4
    train_generator = DataGenerator(
        df=train_df,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        img_shape=IMG_SHAPE,
        max_label_len=max_label_len,
        letters=LETTERS,
        downsample_factor=downsample_factor
    )
    val_generator = DataGenerator(
        df=val_df,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        img_shape=IMG_SHAPE,
        max_label_len=max_label_len,
        letters=LETTERS,
        downsample_factor=downsample_factor
    )
    print("--- 3. Building model ---")
    training_model, prediction_model = build_model(input_shape=IMG_SHAPE, num_chars=NUM_CHARS)
    print(f"--- 4. Compiling model with learning rate: {args.lr} ---")
    training_model.compile(
        optimizer=Adam(learning_rate=args.lr),
        loss={'ctc': lambda y_true, y_pred: y_pred}
    )
    print("--- 5. Setting up callbacks ---")
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    checkpoint = ModelCheckpoint(
        filepath=args.save_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    print(f"--- 6. Starting training for {args.epochs} epochs ---")
    history = training_model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=args.epochs,
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=[checkpoint, early_stopping],
        verbose=1
    )
    print("--- 7. Training complete. ---")
    print(f"Loading best weights from {args.save_path} into prediction model...")
    prediction_model.load_weights(args.save_path)
    pred_model_save_path = args.save_path.replace(".weights.h5", "_prediction.h5")
    prediction_model.save(pred_model_save_path)
    print(f"Best training weights saved to: {args.save_path}")
    print(f"Final prediction model (for inference) saved to: {pred_model_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the CRNN model.")
    parser.add_argument("--data_path", default="sampled_data_100.csv", help="Path to the main data CSV file")
    parser.add_argument("--data_dir", default=".", help="Path to the project's root directory (where data folders are)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for Adam optimizer")
    parser.add_argument("--save_path", default="models/crnn_best.weights.h5", help="Path to save the best model weights")
    args = parser.parse_args()
    main(args)
