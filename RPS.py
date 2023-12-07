# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.
import numpy as np
import random
import time
import os
import tensorflow as tf
import pandas as pd
from tensorflow import keras

def player(prev_play, opponent_history=[], cont=[]):
    """
    A function implementing the logic for a player that returns a play of Rock, Paper, or Scissors represented by the letters R, P, or S. 
    This function utilizes a neural network LSTM with input consisting of the ten previous plays by the player and the next play by the opponent after ten movements by the player.

    Parameters:
    - prev_play (str): The previous play of the opponent.
    - opponent_history (list): Stores the opponent's play history between function calls.
    - cont (list): Stores the position where the opponent's play is stored.

    Returns:
    - str: The play chosen by the player (Rock: 'R', Paper: 'P', Scissors: 'S').
    """
    len_training = 331 # number of plays used to train the model
    play = random.choice(['R', 'P', 'S']) # Initially, the play returned is random.
    while len(opponent_history) == 0: # The first element of prev_play is always empty, so in this case, the first play of the player is stored.
        opponent_history.append(play)
        cont.append(10) # The first opponent's play must be stored in position 10.
        return play
        
    while 0< len(opponent_history) < len_training:
        print('Historial del oponente: \n', opponent_history)
        if len(opponent_history) == cont[0]: #validate if necessary stores opponent play
            opponent_history.append(prev_play)
            opponent_history.append(play)
            c = cont[0]+11 # next position of opponent play
            cont[0] = c
            print('C:   \n', c)
            print('count:  \n', cont[0])
            #play = random.choice(['R', 'P', 'S'])
        else: 
            opponent_history.append(play)
        time.sleep(1)
        print('count:   \n', cont[0])
        return play
    archive_name = 'mi_archivo.txt'
    with open(archive_name, 'w') as archivo:
        #writes each item(play) in the list to a line in the file
        for elemento in opponent_history:
            archivo.write(elemento)

    text = open('mi_archivo.txt', 'rb').read().decode(encoding='utf-8')
           
    vocab = sorted(set(text)) # the unique characteres in text
   
    example_texts = ['abcdefg', 'xyz']
    chars = tf.strings.unicode_split(example_texts, input_encoding='UTF-8') # the text split into tokens first
    #print(chars)
    ids_from_chars = tf.keras.layers.StringLookup(vocabulary=list(vocab), mask_token=None) #convert each character into a numeric ID
    ids = ids_from_chars(chars)
    chars_from_ids = tf.keras.layers.StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)#recover human-readable strings
    chars = chars_from_ids(ids)
    tf.strings.reduce_join(chars, axis=-1).numpy()

    def text_from_ids(ids):
        """join the characters back into strings."""
        return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)
    all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
    ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)#convert the text vector into a stream of character indices.
    # for ids in ids_dataset.take(10):
    #     print(chars_from_ids(ids).numpy().decode('utf-8'))
    seq_length = 10
    sequences = ids_dataset.batch(seq_length+1, drop_remainder=True) #convert these individual characters to sequences of the desired size       
    # for seq in sequences.take(1):
    #     print(chars_from_ids(seq))
    # for seq in sequences.take(5):
    #     print(text_from_ids(seq).numpy())
    
    def split_input_target(sequence):
        """takes a sequence as input, duplicates, and shifts it to align the input and label for each timestep"""
        input_text = sequence[:-1]
        target_text = sequence[1:]
        return input_text, target_text
    
    #split_input_target(list("Tensorflow"))
    dataset = sequences.map(split_input_target)
    # for input_example, target_example in dataset.take(1):
    #     print("Input :", text_from_ids(input_example).numpy())
    #     print("Target:", text_from_ids(target_example).numpy())
    # Batch size
    BATCH_SIZE = 9

    # Buffer size to shuffle the dataset
    # (TF data is designed to work with possibly infinite sequences,
    # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
    # it maintains a buffer in which it shuffles elements).
    BUFFER_SIZE = 100

    dataset = (
        dataset
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE))
    
         
    # Length of the vocabulary in StringLookup Layer
    vocab_size = len(ids_from_chars.get_vocabulary())
    #print('vocab size: ', vocab_size)
    # The embedding dimension
    embedding_dim = 128
    # Number of RNN units
    rnn_units = 1024
    class MyModel(tf.keras.Model):
        def __init__(self, vocab_size, embedding_dim, rnn_units):
            super().__init__(self)
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
            self.gru = tf.keras.layers.GRU(rnn_units,
                                        return_sequences=True,
                                        return_state=True)
            self.dense = tf.keras.layers.Dense(vocab_size)

        def call(self, inputs, states=None, return_state=False, training=False):
            x = inputs
            x = self.embedding(x, training=training)
            if states is None:
                states = self.gru.get_initial_state(x)
            x, states = self.gru(x, initial_state=states, training=training)
            x = self.dense(x, training=training)
            if return_state:
                return x, states
            else:
                return x
            
    #defines the model as a keras.Model subclass
    model = MyModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        rnn_units=rnn_units)
    #example_batch_predictions = 0
    #example_batch_predictions = None
    #print('punto de control')
    #print(dataset.take(1))
    #example_batch_predictions = model(dataset.take(1))

    #Try the model
    for input_example_batch, target_example_batch in dataset.take(1):
        #global example_batch_predictions
        example_batch_predictions = model(input_example_batch)
        #print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

    #sample from the output distribution, to get actual character indices. This distribution is defined by the logits over the character vocabulary.
    sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
    sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()
    #print("Input:\n", text_from_ids(input_example_batch[0]).numpy())
    #print()
    #print("Next Char Predictions:\n", text_from_ids(sampled_indices).numpy())


    #Train the model
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    example_batch_mean_loss = loss(target_example_batch, example_batch_predictions)
    #print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
    #print("Mean loss:        ", example_batch_mean_loss)
    tf.exp(example_batch_mean_loss).numpy()
    model.compile(optimizer='adam', loss=loss)
    # Directory where the checkpoints will be saved
    checkpoint_dir = './training_checkpoints'
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)
    #Execute the training
    EPOCHS = 30
    while len(opponent_history) == len_training:
        history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
        tf.saved_model.save(model, 'Users\DISENOGRAFICO\Downloads\PPT')
        break
    #Generate text    
    class OneStep(tf.keras.Model):
        """The following makes a single step prediction"""
        def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
            super().__init__()
            self.temperature = temperature
            self.model = model
            self.chars_from_ids = chars_from_ids
            self.ids_from_chars = ids_from_chars

            # Create a mask to prevent "[UNK]" from being generated.
            skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
            sparse_mask = tf.SparseTensor(
                # Put a -inf at each bad index.
                values=[-float('inf')]*len(skip_ids),
                indices=skip_ids,
                # Match the shape to the vocabulary
                dense_shape=[len(ids_from_chars.get_vocabulary())])
            self.prediction_mask = tf.sparse.to_dense(sparse_mask)

        @tf.function
        def generate_one_step(self, inputs, states=None):
            # Convert strings to token IDs.
            input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
            input_ids = self.ids_from_chars(input_chars).to_tensor()

            # Run the model.
            # predicted_logits.shape is [batch, char, next_char_logits]
            predicted_logits, states = self.model(inputs=input_ids, states=states,
                                                return_state=True)
            # Only use the last prediction.
            predicted_logits = predicted_logits[:, -1, :]
            predicted_logits = predicted_logits/self.temperature
            # Apply the prediction mask: prevent "[UNK]" from being generated.
            predicted_logits = predicted_logits + self.prediction_mask

            # Sample the output logits to generate token IDs.
            predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
            predicted_ids = tf.squeeze(predicted_ids, axis=-1)

            # Convert from token ids to characters
            predicted_chars = self.chars_from_ids(predicted_ids)

            # Return the characters and model state.
            return predicted_chars, states
    
    #break
    #cont +=1
    try: # Each play, ten previous plays are taken for predict the next move
        restored = tf.saved_model.load('Users\DISENOGRAFICO\Downloads\PPT')
        one_step_model = OneStep(restored, chars_from_ids, ids_from_chars)    
        
        states = None
        
        next_char = tf.constant([text[-10:]])
        
        result = [next_char]
        for n in range(1):
            next_char, states = one_step_model.generate_one_step(next_char, states=states)
            result.append(next_char)
        result = tf.strings.join(result)
         
        play = result[0].numpy().decode('utf-8')
        play = play[-1]
            
       
    except UnboundLocalError:
    #restored = tf.saved_model.load('/tmp/adder')
        print('Modelo no entrenado :(')        
    finally:        
        #print('Jugada Número: ',cont)
        print('Resultado #', len(opponent_history))
        opponent_history.append(play)
        return play