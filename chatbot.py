# Building a Chatbot with Deep NLP

#Importing the Libraries:

import numpy as np
import tensorflow as tf
import re
import time

##### PART 1: DATA PREPROCESSING:-   #####

# Importing the Dataset

lines= open('movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')

conversations= open('movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')

## CREATING A DICTIONARY THAT MAPS EACH LINE WITH ITS ID:

id2line={}

for line in lines:
    
    _line=line.split(' +++$+++ ')
    
    if len(_line)==5:
        
        id2line[_line[0]]= _line[4]
        

## CREATING A LIST OF ALL THE CONVERSATIONS (containing only IDs):

conversations_ids=[]

for conversation in conversations[:-1]:
    
    _conversation=conversation.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    
    conversations_ids.append(_conversation.split(","))
    
    
## GETTING QUESTIONS AND ANSWERS SEPARATELY(to get into input and output format)

questions=[]
answers=[]

for conversation in conversations_ids:
    
    for i in range(len(conversation)-1):
        
        questions.append(id2line[conversation[i]])
        
        answers.append(id2line[conversation[i+1]])
        
## DOING A CLEANING OF THE SENTENCES:
        
def clean_text(text):
    
    text = text.lower()
    
    text=re.sub(r"i'm","i am",text)
    
    text=re.sub(r"he's","he is",text)
    text=re.sub(r"she's","she is",text)
    text=re.sub(r"that's","that is",text)
    text=re.sub(r"what's","what is",text)
    text=re.sub(r"\'ll"," will",text)
    text=re.sub(r"\'ve"," have",text)
    text=re.sub(r"\'re"," are",text)
    text=re.sub(r"\'d"," would",text)
    text=re.sub(r"where's","where is",text)
    text=re.sub(r"won't","will not",text)
    text=re.sub(r"can't","cannot",text)
    text=re.sub(r"would'nt","would not",text)
    text=re.sub(r"could'nt","could not",text)
    text=re.sub(r"don't","do not",text)
    text=re.sub(r"didn't","did not",text)
    text=re.sub(r"there's","there is",text)
    text=re.sub(r"\'til","till",text)
    text=re.sub(r"wasn't","was not",text)
    text=re.sub(r"bout","about",text)
    text=re.sub(r"it's","it is",text)
    
    text=re.sub(r"[()-=+*&^%$#@{}\"|':;><,.?/]","",text)
                  
    return text
    
## CLEANING THE QUESTIONS:
    
clean_questions=[]

for question in questions:
    
    clean_questions.append(clean_text(question))
    
## SIMILARLY CLEANING THE ANSWERS:
    
clean_answers=[]

for answer in answers:
    
    clean_answers.append(clean_text(answer))
    
## CREATING A DICTIONARY THAT MAPS EACH WORD TO ITS OCCURENCE:

word2count= {}
for question in clean_questions:
    
    for word in question.split():
        
        if word not in word2count:
            
            word2count[word]=1
            
        else:
            word2count[word]+=1
    
for answer in clean_answers:
    
    for word in answer.split():
        
        if word not in word2count:
            
            word2count[word]=1
            
        else:
            word2count[word]+=1
    
    
## CREATING TWO DICTIONARIES THAT MAPS ANSWER AND QUESTION WORDS TO A UNIQUE INTEGER:

threshold=20
            
question2int={}
word_number=0

for word,count in word2count.items():
    
    if count >= threshold:
        
        question2int[word]=word_number
        word_number+=1
        
answer2int={}
word_number=0

for word,count in word2count.items():
    
    if count >= threshold:
        
        answer2int[word]=word_number
        word_number+=1
        
        
## ADDING LAST TOKENS LEFT TO THE LAST OF BOTH THE ABOVE DICTIONARIES:
        
tokens=['<PAD>','<EOS>','<OUT>','<SOS>']

for token in tokens:
    
    question2int[token]=len(question2int)+1
    answer2int[token]=len(answer2int)+1
    
        
## INVERSING THE answer2int DICTIONARY (required for training purpose)
    
answersint2word={w_i:w for w,w_i in answer2int.items()}


## ADDING THE END OF STRING TOKENTO THE END OF EVERY ANSWER:

for i in range(len(clean_answers)):
    
    clean_answers[i] += " <EOS>"
    
    
## TRANSLATING ALL THE QUESTIONS AND ANSWERS INTO INTEGERS :
questions_to_int= []

for question in clean_questions:

    ints=[]
    
    for word in question.split():
        
        if word in question2int:
        
            ints.append(question2int[word])
        else:
            
            ints.append(question2int["<OUT>"])
    
    questions_to_int.append(ints) 

    
answers_to_int= []

for answer in clean_answers:

    ints=[]
    
    for word in answer.split():
        
        if word in answer2int:
        
            ints.append(answer2int[word])
        else:
            
            ints.append(answer2int["<OUT>"])
    
    answers_to_int.append(ints) 
  
## SORTING QUESTIONS AND ANSWERS BY THE LENGTH OF QUESTIONS:
    
sorted_clean_questions=[]
sorted_clean_answers=[]

for length in range(1,25+1):
    
    for i in enumerate(questions_to_int):
        
        if len(i[1])==length:
            
            sorted_clean_questions.append(questions_to_int[i[0]])
            sorted_clean_answers.append(answers_to_int[i[0]])
            
        
    ###############    END   ##############################
    
    
#####   PART 2: BUILDING THE SEQ2SEQ MODEL ############
    
## CREATING TENSORFLOW PLACEHOLDER FOR INPUT,TARGET,LEARNING RATE,DROPOUT RATE:
    
def model_inputs():
    
    inputs =tf.placeholder(tf.int32, [None,None], name="input")
    
    targets = tf.placeholder(tf.int32,[None,None], name="target")
    
    lr= tf.placeholder(tf.float32, name="learning_rate")

    keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    return inputs, targets, lr, keep_prob    


    
### PREPROCESSING THE TARGETS:
    
def preprocess_targets(targets, word2int, batch_size):
    
    left_side=tf.fill([batch_size,1], word2int["<SOS>"])
    
    right_side= tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1])
    
    preprocessed_targets= tf.concat([left_side,right_side],1)
    
    return preprocessed_targets


## CREATING THE ENCODER RNN LAYER:
    
def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
   
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    
    encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
                                                                    cell_bw = encoder_cell,
                                                                    sequence_length = sequence_length,
                                                                    inputs = rnn_inputs,
                                                                    dtype = tf.float32)
    return encoder_state

# DECODING THE TRAINING SET:
    
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau", num_units = decoder_cell.output_size)
    
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              name = "attn_dec_train")
    
    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                              training_decoder_function,
                                                                                                              decoder_embedded_input,
                                                                                                              sequence_length,
                                                                                                              scope = decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    
    return output_function(decoder_output_dropout)


##  DECODING THE TEST/ VALIDATION SET:
    
def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, num_words, decoding_scope, output_function, keep_prob, batch_size):
    
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau", num_units = decoder_cell.output_size)
    
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                              encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              decoder_embeddings_matrix,
                                                                              sos_id,
                                                                              eos_id,
                                                                              maximum_length,
                                                                              num_words,
                                                                              name = "attn_dec_inf")
    
    test_predictions, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                                test_decoder_function,
                                                                                                                scope = decoding_scope)
    return test_predictions

# Creating the Decoder RNN
def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
   
    with tf.variable_scope("decoding") as decoding_scope:
    
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        
        weights = tf.truncated_normal_initializer(stddev = 0.1)
        
        biases = tf.zeros_initializer()
        
        output_function = lambda x: tf.contrib.layers.fully_connected(x,
                                                                      num_words,
                                                                      None,
                                                                      scope = decoding_scope,
                                                                      weights_initializer = weights,
                                                                      biases_initializer = biases)
        training_predictions = decode_training_set(encoder_state,
                                                   decoder_cell,
                                                   decoder_embedded_input,
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


# Building the seq2seq model
def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questionswords2int):
   
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                              answers_num_words + 1,
                                                              encoder_embedding_size,
                                                              initializer = tf.random_uniform_initializer(0, 1))
    
    encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
    
    preprocessed_targets = preprocess_targets(targets, questionswords2int, batch_size)
    
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words + 1, decoder_embedding_size], 0, 1))
    
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input,
                                                         decoder_embeddings_matrix,
                                                         encoder_state,
                                                         questions_num_words,
                                                         sequence_length,
                                                         rnn_size,
                                                         num_layers,
                                                         questionswords2int,
                                                         keep_prob,
                                                         batch_size)
    return training_predictions, test_predictions

# Setting the Hyperparameters

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
    

# Defining a session

tf.reset_default_graph()

session = tf.InteractiveSession()

# Loading the model inputs
inputs, targets, lr, keep_prob = model_inputs()

# Setting the sequence length
sequence_length = tf.placeholder_with_default(25, None, name = 'sequence_length')

# Getting the shape of the inputs tensor
input_shape = tf.shape(inputs)

## GETTING THE TRAINING AND TEST PREDICTIONS:


training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs,[-1]),
                                                       targets, 
                                                       keep_prob, 
                                                       batch_size, 
                                                       sequence_length, 
                                                       len(answer2int), 
                                                       len(question2int),
                                                       encoding_embedding_size, 
                                                       decoding_embedding_size, 
                                                       rnn_size,
                                                       num_layers, 
                                                       question2int)


# Setting up the Loss Error, the Optimizer and Gradient Clipping

with tf.name_scope("optimization"):
    
    loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions,
                                                  targets,
                                                  tf.ones([input_shape[0], sequence_length]))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    
    gradients = optimizer.compute_gradients(loss_error)
    
    clipped_gradients = [(tf.clip_by_value(grad_tensor, -5., 5.), grad_variable) for grad_tensor, grad_variable in gradients if grad_tensor is not None]
    
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)

    
  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    