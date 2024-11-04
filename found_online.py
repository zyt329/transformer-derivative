import time
from tqdm import tqdm
import pickle
import re

import tensorflow as tf
from keras.preprocessing.text import Tokenizer
import numpy as np


""" I. Data preparation """
def split(txt, seps):
    default_sep = seps[0]

    # we skip seps[0] because that's the default separator
    for sep in seps[1:]:
        txt = txt.replace(sep, default_sep)
    return [i.strip() for i in txt.split(default_sep)]

def replace_with_x(functions, var_name):
    regex = '(exp\^|\+|\-|\(|\)|\^|sin|cos|\*)'
    function_splitted = re.split(regex, functions)
    e = 0
    const_re = re.compile('(?!x)[a-zA-Z]')
    while e < len(function_splitted):
        if function_splitted[e] not in ['sin', 'cos', 'exp^']:
            function_splitted[e] = function_splitted[e].replace(var_name, 'x')
            consts = const_re.findall(function_splitted[e])
            for c in consts:
                function_splitted[e] = function_splitted[e].replace(c, 'const')
        if function_splitted[e] in ["", "\n"]:
            function_splitted.pop(e)
        e += 1
    return function_splitted

def extract_string_expresisons(path='train.txt'):
    file = open(path)

    fxs = []
    dfxs = []
    
    for i in file:
        i = str(i).replace("\n", "")
        
        (org, derv) = i.split("=")
        var_name = org[-1]
        
        listed_org = replace_with_x(org[2:-4], var_name)
        fxs.append(" ".join(listed_org))
        
        listed_derv = replace_with_x(derv, var_name)
        dfxs.append(" ".join(listed_derv))

    data_opt = []
    data_ipt  = []

    for i in tqdm(range(len(fxs))):
        f = fxs[i]
        df = dfxs[i]

        data_opt.append(df)
        data_ipt.append(f)

    return data_ipt, data_opt


"""
    II. Tokenizer:
 - implement tokenizer that returns ids by mapping words found in string with their indexes in vocab
 - text is separated by whitespaces, vocab. tokenize by space then locate each token in vocab

"""

def train_tokenizer(train_examples):
    tokenizer_ipt = Tokenizer(oov_token='[UNK]', filters='')
    tokenizer_opt = Tokenizer(oov_token='[UNK]', filters='')

    tokenizer_ipt.fit_on_texts((ipt.numpy().decode('utf8') for (ipt, opt) in train_examples))
    tokenizer_opt.fit_on_texts((opt.numpy().decode('utf8') for (ipt, opt) in train_examples))

    _save_pickle('tokenizers/tokenizer_ipt.pkl', tokenizer_ipt)
    _save_pickle('tokenizers/tokenizer_opt.pkl', tokenizer_opt)
    return tokenizer_ipt, tokenizer_opt

def _save_pickle(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def _load_pickle(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj

""" IV. Define modules: attention functions, point wise feed forward network, decoder layer, and the decoder itself"""

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model) # shape (position, d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32) # shape: (position, d_model)

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def scaled_dot_product_attention(q, k, v, mask):

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32) # depth
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        
        self.dense = tf.keras.layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)
        
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, 
                                    (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
            
        return output, attention_weights

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)
    
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)
        
    
    def call(self, x, training, dec_inp_padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, dec_inp_padding_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
        
        attn2, attn_weights_block2 = self.mha2(out1, out1, out1, dec_inp_padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)
        
        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)
        
        return out3, attn_weights_block1, attn_weights_block2

class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) 
                        for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
    
    def call(self, x, training, dec_inp_padding_mask):

        seq_len = tf.shape(x)[1]
        # attention_weights = {}
        
        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, training, dec_inp_padding_mask)
                                                    
        #   attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
        #   attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
        
        # x.shape == (batch_size, target_seq_len, d_model)
        return x

""" V. Create Transformer model"""
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
                 target_vocab_size, pe_input, rate=0.1):
        super(Transformer, self).__init__()
        
        # self.encoder = Encoder(num_layers, d_model, num_heads, dff, 
        #                        input_vocab_size, pe_input, rate)
        
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, 
                               input_vocab_size, pe_input, rate)
        

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
        
    def call(self, inp, training):
        # Keras models prefer if you pass all your inputs in the first argument

        dec_inp_padding_mask = self.create_masks(inp)

        # print('enc_padding_mask: ', enc_padding_mask)
        # enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output = self.decoder(inp, training, dec_inp_padding_mask)

        
        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output
    
    def create_masks(self, inp):
        # Encoder padding mask
        # enc_padding_mask = create_padding_mask(inp)

        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        dec_inp_padding_mask = create_padding_mask(inp)

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by 
        # the decoder.
        # look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        # dec_target_padding_mask = create_padding_mask(tar)
        # look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return dec_inp_padding_mask


# Loss and Accuracy metrics

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps
        
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

if __name__ == "__main__":
    data_ipt, data_opt = extract_string_expresisons('../input/derivative-train-set/train.txt')

    num_layers = 2
    d_model = 32
    dff = 1024
    num_heads = 4

    dropout_rate = 0.05
    EPOCHS = 10
    BATCH_SIZE = 512
    MAX_LENGTH = 30

    train_examples = tf.data.Dataset.from_tensor_slices((data_ipt, data_opt))

    tokenizer_ipt, tokenizer_opt = train_tokenizer(train_examples)
    
    input_vocab_size = len(tokenizer_ipt.get_config()["word_index"])
    target_vocab_size = len(tokenizer_opt.get_config()["word_index"])

    def encode(ipt, opt):
        ipt = tokenizer_ipt.texts_to_sequences([ipt.numpy().decode('utf8')])[0]
        opt = tokenizer_opt.texts_to_sequences([opt.numpy().decode('utf8')])[0]
        return ipt, opt

    def tf_encode(ipt, opt):
        result_ipt, result_opt = tf.py_function(encode, [ipt, opt], [tf.int64, tf.int64])
        result_ipt.set_shape([None])
        result_opt.set_shape([None])
        return result_ipt, result_opt

    def filter_max_length(x, y, max_length=MAX_LENGTH):
        return tf.logical_and(tf.size(x) <= max_length,
                                tf.size(y) <= max_length)

    train_dataset = train_examples.map(tf_encode)
    train_dataset = train_dataset.filter(filter_max_length)
    # cache the dataset to memory to get a speedup while reading from it.
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.padded_batch(BATCH_SIZE, padded_shapes=([MAX_LENGTH], [MAX_LENGTH]))

    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    learning_rate = CustomSchedule(d_model)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                        epsilon=1e-9)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    transformer = Transformer(num_layers, d_model, num_heads, dff,
                            input_vocab_size, 
                            target_vocab_size,
                            pe_input=input_vocab_size, 
                            rate=dropout_rate)
    
    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.int64), tf.TensorSpec(shape=(None, None), dtype=tf.int64)])
    def train_step(inp, tar):
        tar_real = tar
        with tf.GradientTape() as tape:
            predictions = transformer(inp, training = True)
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)    
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_loss(loss)
        train_accuracy(accuracy_function(tar_real, predictions))

    # checkpoint_path = "ckpt\\"

    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)

    # ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # # if a checkpoint exists, restore the latest checkpoint.
    # if ckpt_manager.latest_checkpoint:
    #     ckpt.restore(ckpt_manager.latest_checkpoint)
    #     print ('Latest checkpoint restored!!', ckpt_manager.latest_checkpoint)

    max_acc = 0

    for epoch in range(EPOCHS):
        start = time.time()
        
        train_loss.reset_states()
        train_accuracy.reset_states()
        
        early_stop = 0
        for (batch, (inp, tar)) in enumerate(train_dataset):
            train_step(inp, tar)
        
            if batch % 10 == 0:
                print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                    epoch + 1, batch, train_loss.result(), train_accuracy.result()))
            
                if train_accuracy.result() - max_acc > 0:
                    max_acc = train_accuracy.result()
                    early_stop = 0
                else:
                    early_stop += 1
                if early_stop > 10:
                    break
        if early_stop > 5:
            ckpt_save_path = ckpt.save("ckpt/checkpoint-lr")
            print ('Early stop!!! Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))
            transformer.summary()
            break
    
        ckpt_save_path = ckpt.save("ckpt/checkpoint-lr")
        print ('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))
        
        print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
                                                    train_loss.result(), 
                                                    train_accuracy.result()))

        print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))