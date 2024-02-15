import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load preprocessed data
with open('english_data.txt', 'r', encoding='utf-8') as file:
    english_sentences = file.readlines()

with open('sinhala_data.txt', 'r', encoding='utf-8') as file:
    sinhala_sentences = file.readlines()

# Preprocess data (tokenization, padding, etc.)
english_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
english_tokenizer.fit_on_texts(english_sentences)
english_seq = english_tokenizer.texts_to_sequences(english_sentences)
english_padded_seq = tf.keras.preprocessing.sequence.pad_sequences(english_seq, padding='post')

sinhala_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
sinhala_tokenizer.fit_on_texts(sinhala_sentences)
sinhala_seq = sinhala_tokenizer.texts_to_sequences(sinhala_sentences)
sinhala_padded_seq = tf.keras.preprocessing.sequence.pad_sequences(sinhala_seq, padding='post')

# Placeholder for preprocessed data
encoder_input_data = english_padded_seq
decoder_input_data = sinhala_padded_seq[:, :-1]
decoder_target_data = sinhala_padded_seq[:, 1:]

# Define the encoder-decoder model
# Define encoder
encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(input_dim=len(english_tokenizer.word_index) + 1, output_dim=256)(encoder_inputs)
encoder_lstm = LSTM(units=512, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Define decoder
decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(input_dim=len(sinhala_tokenizer.word_index) + 1, output_dim=256)(decoder_inputs)
decoder_lstm = LSTM(units=512, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(len(sinhala_tokenizer.word_index) + 1, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model
# model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# # Compile the model
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# # Train the model
# model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
#           batch_size=64,
#           epochs=10,
#           validation_split=0.2)

# # Save the model
# model.save('translation_model.h5')

# Load the trained model
model = tf.keras.models.load_model('translation_model.h5')

# Function to translate English sentence to Sinhala
def translate_sentence(sentence):
    # Tokenize the input sentence
    input_tokens = english_tokenizer.texts_to_sequences([sentence])
    input_tokens = pad_sequences(input_tokens, padding='post')

    # Define decoder input (initially filled with zeros)
    decoder_input = np.zeros((len(input_tokens), 1))

    # Translate the input sentence
    translated_tokens = model.predict([input_tokens, decoder_input])

    # Convert the translated tokens back to words
    translated_words = []
    for token in translated_tokens:
        token_index = np.argmax(token)
        if token_index != 0 and token_index in sinhala_tokenizer.index_word:
            word = sinhala_tokenizer.index_word[token_index]
            if word != '<end>':
                translated_words.append(word)
            else:
                break

    # Convert the list of words to a sentence
    translated_sentence = ' '.join(translated_words)

    return translated_sentence

# Example usage
english_sentence = "you will receive a package in the mail"
sinhala_translation = translate_sentence(english_sentence)
print("Sinhala translation:", sinhala_translation)
