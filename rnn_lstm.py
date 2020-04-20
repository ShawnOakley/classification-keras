from keras.models import models
from keras.layers import Input, LSTM, Dense
from keras.utils import plot_model
from keras.utils import plot_model
# Debugging and visualizations
from keras.callbacks import Tensorboard
# used to save progress
from keras.callbacks import ModelCheckpoint

batch_size = 64
epochs = 100
latent_dimension = 256
num_samples = 1000

input_texts = []
target_texts = []

input_characters = set()
target_characters = set()

with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('/n')

# extract english sentence and french translation
for line in lines[: min(num_samples, len(lines) - 1)]:
    # split line by tab
    input_text, target_text = line.split('\t')
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)

    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

# Generate sorted lists for use in one hot encoding
input_characters = sorted(list(input_characters))

target_characters = sorted(list(target_characters))

# Generate number of characters per list
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)

# Find longest english sentence
max_encoder_seq_length = max([len(txt) for txt in input_texts])

# Find longest french sentence
max_decoder_seq_length = max([len(txt) for txt in target_texts])

# Maps english character to integer index
input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])

# Maps french character to integer index
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])

# batch_size, max_english_sentence_length, max_english_characters
encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')

# batch_size, max_french_sentence_length, max_french_characters
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

# Loop through data and set one-hot encoded elements
for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    # t is the time dimension
    for t, char in enumerate(input_text):
        # Set the one-hot encoded character for english
        encoder_input_data[i, t, input_token_index[char]] = 1.

        for t, char in enumerate(target_text):
        # Set the one-hot encoded character for input for french 
        decoder_input_data[i, t, target_token_index[char]] = 1.
        # Encode for prior time period output if t is greater than 1
        # Other dataset used for training
        if t > 0:
            decoder_target_data(i, t - 1, target_token_index(char)) = 1.

# Input is a sequence of one-hot encoded English characters in a sentence
encoder_inputs = Input(shape=None, num_encoder_tokens)

# Return = true gives access to hidden and cell states
# hidden state is for long-term memory
# cell state is for short term memory
encoder = LSTM(latent_dimension, return_state=True)

# get back outputs, hidden state, and cell state from invoking encoder
encoder_outputs, state_h, state_c = encoder(encoder_inputs)

encoder_states = (state_h, state_c)

decoder_inputs = Input(shape=(None, num_decoder_tokens))

decoder_ltsm = LTSM(latent_dimension, return_sequences=True, return_state=True)
# Predicted states of the french characters
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                    # Encoder states are english sentences
                                    initial_state=encoder_states)

decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder, inputs, decoder_inputs], decoder_inputs)

# Plot result
plot_model(model, to_file='model.png', show_shapes=True)

# RMSPROP for RNNs, Categorical cross-entropy b/c it's a classification problem
# Since we're identifying the appropriate character from a group
model.compile(optimizer='rmsprops', loss='categorical_crossentropy')

# Set up a checkpoint to save model everytime we reach a new min
filepath = 'weights.best.hdf5'

checkpoint - ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2,
            callbacks=[TensorBoard(log_dir='/tmp/autoencoder').checkpoint])
