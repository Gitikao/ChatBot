# Building a ChatBot with Deep NLP
import numpy as np
import tensorflow as tf
import re
import time
#import keras 
#from tensorflow.keras.layers import Dense,Dropout, LSTM
#from tensorflow.keras.models import Sequential

print(tf.__version__)
lines = open('movie_lines.txt', encoding='utf-8',
             errors='ignore').read().split('\n')
conversations = open('movie_conversations.txt',
                     encoding='utf-8', errors='ignore').read().split('\n')
id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]
conversations_ids = []
for conversation in conversations[:-1]:
    temp1 = conversation.split(
        ' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
    conversations_ids.append(temp1.split(','))
questions = []
answers = []
for conversation in conversations_ids:
    for i in range(len(conversation) - 1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])


def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"'ve", "have", text)
    text = re.sub(r"'re", "are", text)
    text = re.sub(r"won't", "would not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text)
    return text


clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))
clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))
word2count = {}
for question in clean_questions:
    for word in question.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1
for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1
threshold = 20
questionwords2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold:
        questionwords2int[word] = word_number
        word_number += 1

answerwords2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold:
        answerwords2int[word] = word_number
        word_number += 1

tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
for token in tokens:
    questionwords2int[token] = len(questionwords2int) + 1
for token in tokens:
    answerwords2int[token] = len(answerwords2int) + 1

answerint2word = {w_i: w for w, w_i in answerwords2int.items()}

for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>'

questions_to_int = []
for question in clean_questions:
    ints = []
    for word in question.split():
        if word not in questionwords2int:
            ints.append(questionwords2int['<OUT>'])
        else:
            ints.append(questionwords2int[word])
    questions_to_int.append(ints)

answers_to_int = []
for answer in clean_answers:
    ints = []
    for word in answer.split():
        if word not in answerwords2int:
            ints.append(answerwords2int['<OUT>'])
        else:
            ints.append(answerwords2int[word])
    answers_to_int.append(ints)

sorted_clean_questions = []
sorted_clean_answers = []
for length in range(1, 26):
    for i in enumerate(questions_to_int):
        if len(i[1]) == length:
            sorted_clean_questions.append(questions_to_int[i[0]])
            sorted_clean_answers.append(answers_to_int[i[0]])


def model_inputs():
  
    tf.compat.v1.disable_eager_execution()


    inputs = tf.compat.v1.placeholder(tf.int32, [None, None], name = 'input')
    targets = tf.compat.v1.placeholder(tf.int32, [None, None], name = 'target')
    lr = tf.compat.v1.placeholder(tf.float32, name = 'learning_rate')
    keep_prob = tf.compat.v1.placeholder(tf.float32, name = 'keep_prob')

  # lr = tf.placeholder(tf.float32, name = 'learning_rate')
  #  keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
    return inputs, targets, lr, keep_prob
 
def preprocess_targets(targets, word2int, batch_size):
    left_side = tf.fill([batch_size, 1], word2int['<SOS>'])
    right_side = tf.strided_slice(targets, [0, 0], [batch_size, -1], [1, 1])
    preprocessed_targets = tf.concat([left_side, right_side], axis=1)
    return preprocessed_targets


def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(
        lstm, input_keep_prob=keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    _, encoder_state = tf.nn.bidirectional_rnn(
        cell_fw=encoder_cell, cell_bw=encoder_cell, sequence_length=sequence_length, inputs=rnn_inputs, dtype=tf.float32)

    return encoder_state


def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(
        attention_states, attention_options='bahdanau', num_units=decoder_cell.output.size)
    training_decoder_function = tf.contrib.seq2sec.attention_decoder_fn_train(
        encoder_state[0], attention_keys, attention_values, attention_score_function, attention_construct_function, name="atten_dec_train")
    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(
        decoder_cell, training_decoder_function, decoder_embedded_input, sequence_length, scope=decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)


def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, num_words, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(
        attention_states, attention_options='bahdanau', num_units=decoder_cell.output.size)
    test_decoder_function = tf.contrib.seq2sec.attention_decoder_fn_inference(output_function, encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              decoder_embeddings_matrix,
                                                                              sos_id,
                                                                              eos_id,
                                                                              maximum_length,
                                                                              num_words,
                                                                              name="atten_dec_inf")

    test_predictions, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                                test_decoder_function,
                                                                                                                scope=decoding_scope)
    return test_predictions


def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(
            lstm, input_keep_prob=keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        weights = tf.truncated_normal_initializer(stddev=0.1)
        biases = tf.zeros_initializer()

        def output_function(x): return tf.contrib.layers.fully_connected(x,
                                                                         num_words,
                                                                         None,
                                                                         scope=decoding_scope,
                                                                         weights_initializer=weights,
                                                                         biases_initializer=biases)
        training_predictions = decode_training_set(encoder_state, decoder_cell, decoder_embedded_input,
                                                   sequence_length,
                                                   decoding_scope,
                                                   output_function,
                                                   keep_prob,
                                                   batch_size)
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state,
                                           decoder_cell,
                                           decoder_embeddings_matrix,
                                           word2int['<SOS>'],
                                           word2int['<EOS>'],
                                           sequence_length - 1,
                                           num_words,
                                           decoding_scope,
                                           output_function,
                                           keep_prob,
                                           batch_size)
        return training_predictions, test_predictions


def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questionwords2int):
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                              answers_num_words+1,
                                                              encoder_embedding_size,
                                                              initializer=tf.random_uniform_initializer(0, 1))
    encoder_state = encoder_rnn(
        encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
    preprocessed_targets = preprocess_targets(
        targets, questionwords2int, batch_size)
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform(
        [questions_num_words + 1, decoder_embedding_size], 0, 1))
    decoder_embedded_input = tf.nn.embedding_lookup(
        decoder_embeddings_matrix, preprocessed_targets)
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input,
                                                         decoder_embeddings_matrix,
                                                         encoder_state,
                                                         questions_num_words,
                                                         sequence_length,
                                                         rnn_size,
                                                         num_layers,
                                                         questionwords2int,
                                                         keep_prob,
                                                         batch_size)
    return training_predictions, test_predictions


epochs = 100
batch_size = 64
rnn_size = 512
num_layers = 3
encoding_embedding_size = 512
decoding_embedding_size = 512
learning_rate = 0.01
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.5


from tensorflow.python.framework import ops
ops.reset_default_graph()
sess=tf.compat.v1.InteractiveSession()

inputs, targets, lr, keep_prob = model_inputs()

sequence_length = tf.compat.v1.placeholder_with_default(25, None, name = 'sequence_length')

input_shape = tf.shape(inputs )

training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs, [-1]),
                                                       targets,
                                                       keep_prob,
                                                       batch_size,
                                                       sequence_length,
                                                       len(answerwords2int),
                                                       len(questionwords2int),
                                                       encoding_embedding_size,
                                                       decoding_embedding_size,
                                                       rnn_size,
                                                       num_layers,
                                                       questionwords2int)

