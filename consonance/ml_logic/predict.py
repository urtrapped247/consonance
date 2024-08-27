import os
import numpy as np
import tensorflow as tf
import keras
from keras import ops
from keras import layers
import argparse

# To run, be sure to include the model to load and the image to run a prediction on
# python predict.py --model /Users/ninjamac/code/urtrapped247/Consonance/data/predict_dual_1k.keras
# --image /Users/ninjamac/code/DataSets/test/images_cropped/music_0.png


os.environ["KERAS_BACKEND"] = "tensorflow"

def preprocess_single_image(image_path):
    """Preprocess a single image for model prediction."""
    img_width = 350
    img_height = 50
    img = tf.io.read_file(image_path)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = ops.image.resize(img, [img_height, img_width])
    img = ops.transpose(img, axes=[1, 0, 2])
    img = tf.expand_dims(img, axis=0)
    return img

def predict_single_image(model, image_path):
    """Predict the musical notes from a single image."""
    img = preprocess_single_image(image_path)
    preds = model.predict(img)
    pred_texts = decode_batch_predictions(preds)
    return pred_texts[0]

def decode_batch_predictions(pred):
    combined_dict = {
        'B3_whole': 0, 'C4_whole': 1, 'D4_whole': 2, 'E4_whole': 3, 'F4_whole': 4, 'G4_whole': 5, 'A4_whole': 6, 'B4_whole': 7, 'C5_whole': 8,
        'D5_whole': 9, 'E5_whole': 10, 'F5_whole': 11, 'G5_whole': 12, 'A5_whole': 13, 'B5_whole': 14, 'C6_whole': 15, 'D6_whole': 16,
        'B3_half': 17, 'C4_half': 18, 'D4_half': 19, 'E4_half': 20, 'F4_half': 21, 'G4_half': 22, 'A4_half': 23, 'B4_half': 24, 'C5_half': 25,
        'D5_half': 26, 'E5_half': 27, 'F5_half': 28, 'G5_half': 29, 'A5_half': 30, 'B5_half': 31, 'C6_half': 32, 'D6_half': 33,
        'B3_quarter': 34, 'C4_quarter': 35, 'D4_quarter': 36, 'E4_quarter': 37, 'F4_quarter': 38, 'G4_quarter': 39, 'A4_quarter': 40, 'B4_quarter': 41, 'C5_quarter': 42,
        'D5_quarter': 43, 'E5_quarter': 44, 'F5_quarter': 45, 'G5_quarter': 46, 'A5_quarter': 47, 'B5_quarter': 48, 'C6_quarter': 49, 'D6_quarter': 50,
        'B3_eighth': 51, 'C4_eighth': 52, 'D4_eighth': 53, 'E4_eighth': 54, 'F4_eighth': 55, 'G4_eighth': 56, 'A4_eighth': 57, 'B4_eighth': 58, 'C5_eighth': 59,
        'D5_eighth': 60, 'E5_eighth': 61, 'F5_eighth': 62, 'G5_eighth': 63, 'A5_eighth': 64, 'B5_eighth': 65, 'C6_eighth': 66, 'D6_eighth': 67,
        'B3_16th': 68, 'C4_16th': 69, 'D4_16th': 70, 'E4_16th': 71, 'F4_16th': 72, 'G4_16th': 73, 'A4_16th': 74, 'B4_16th': 75, 'C5_16th': 76,
        'D5_16th': 77, 'E5_16th': 78, 'F5_16th': 79, 'G5_16th': 80, 'A5_16th': 81, 'B5_16th': 82, 'C6_16th': 83, 'D6_16th': 84
    }

    int_to_label = {v: k for k, v in combined_dict.items()}

    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = ctc_decode(pred, input_length=input_len, greedy=True)[0][0]

    output_text = []
    for res in results:
        label_sequence = [int_to_label.get(int(i), '') for i in res.numpy()]  
        label_string = ' '.join(label_sequence)
        output_text.append(label_string)
    return output_text

def ctc_decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1):
    input_shape = ops.shape(y_pred)
    num_samples, num_steps = input_shape[0], input_shape[1]
    y_pred = ops.log(ops.transpose(y_pred, axes=[1, 0, 2]) + keras.backend.epsilon())
    input_length = ops.cast(input_length, dtype="int32")

    if greedy:
        (decoded, log_prob) = tf.nn.ctc_greedy_decoder(
            inputs=y_pred, sequence_length=input_length
        )
    else:
        (decoded, log_prob) = tf.compat.v1.nn.ctc_beam_search_decoder(
            inputs=y_pred,
            sequence_length=input_length,
            beam_width=beam_width,
            top_paths=top_paths,
        )
    decoded_dense = []
    for st in decoded:
        st = tf.SparseTensor(st.indices, st.values, (num_samples, num_steps))
        decoded_dense.append(tf.sparse.to_dense(sp_input=st, default_value=-1))
    return (decoded_dense, log_prob)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict musical notes from a single image.")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model file.")
    parser.add_argument("--image", type=str, required=True, help="Path to the image file.")
    args = parser.parse_args()

    # Load the model
    prediction_model = keras.models.load_model(args.model)

    # Predict the notes from the image
    predicted_notes = predict_single_image(prediction_model, args.image)

    # Print the result
    print(f"Predicted Notes: {predicted_notes}")
