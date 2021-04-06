from keras.layers import Input, Lambda, Dense
from keras import backend as K
from keras.layers import Input, Softmax, Embedding, Add, Multiply, Lambda, Dense, Masking
from keras_layer_normalization import LayerNormalization
from keras_transformer import get_encoders
from keras_bert.backend import keras
from keras_bert.activations import gelu
from model.transformer_util.extras import TiedOutputEmbedding
from keras_bert.optimizers import AdamWarmup
from keras import regularizers
from keras.optimizers import Adam
import tensorflow as tf

from model.multihot_utils import ReusableEmbed_Multihot
from model.LossFunction import batch_crossentropy, confidence_penalty, recall_at_10


def BERT(config):
    num_times = config.num_times
    num_input_0 = config.num_input_0
    num_input_1 = config.num_input_1
    num_input_2 = config.num_input_2
    num_input_3 = config.num_input_3
    num_input_4 = config.num_input_4
    
    num_layers = config.num_layers
    embedding_dim = config.embedding_dim
    num_heads = config.num_heads
    
    transformer_dropout = config.transformer_dropout
    embedding_dropout = config.embedding_dropout
    l2_reg_penalty_weight = config.l2_reg_penalty_weight
    confidence_penalty_weight = config.confidence_penalty_weight
    lrate = config.lrate
    
    feed_forward_dim = 4 * embedding_dim
    attention_activation=None
    feed_forward_activation=gelu
    training=True
    trainable=training

    input_0 = Input(shape=(num_times, num_input_0), dtype='float', name='course_ids')
    input_1 = Input(shape=(num_times, num_input_1), dtype='float', name='major_ids')
    input_2 = Input(shape=(num_times, num_input_2), dtype='float', name='relative_ids')
    input_3 = Input(shape=(num_times, num_input_3), dtype='float', name='name_ids')
    input_4 = Input(shape=(num_times, num_input_4), dtype='float', name='num_samples')
    
    target = Input(shape=(num_times, num_input_0), dtype='float', name='target')
    use_pred = Input(shape=(num_times, ), dtype='float', name='use_pred')
    
    l2_reg = (regularizers.l2(l2_reg_penalty_weight) if l2_reg_penalty_weight else None)
    
    embedding_0, embedding_matrix = ReusableEmbed_Multihot(
            num_input_0, embedding_dim, input_length=num_times,name='course_embeddings_0',embeddings_regularizer=l2_reg)(input_0)
    
    embedding_1 = ReusableEmbed_Multihot(num_input_1, embedding_dim, input_length=num_times, name='relative_embedding')(input_1)[0]
    embedding_2 = ReusableEmbed_Multihot(num_input_2, embedding_dim, input_length=num_times, name='major_embeddings')(input_2)[0]
    embedding_3 = ReusableEmbed_Multihot(num_input_3, embedding_dim, input_length=num_times, name='name_embedding')(input_3)[0]
    embedding_4 = ReusableEmbed_Multihot(num_input_4, embedding_dim, input_length=num_times, name='num_embedding')(input_4)[0]
    
    next_step_input = Add(name='Embedding_Add')([embedding_0, embedding_1, embedding_2, embedding_3, embedding_4])

    next_step_input = get_encoders(
        encoder_num=num_layers,
        input_layer=next_step_input,
        head_num=num_heads,
        hidden_dim=feed_forward_dim,
        attention_activation=attention_activation,
        feed_forward_activation=feed_forward_activation,
        dropout_rate=transformer_dropout)

    predict = Softmax(name='word_predictions_0')(TiedOutputEmbedding(
        projection_regularizer=l2_reg,
        projection_dropout=embedding_dropout,
        name='word_prediction_logits_1')([next_step_input, embedding_matrix]))

    loss_layer_1 = Lambda(lambda x: batch_crossentropy(*x), name="loss_layer_1")([target, predict, use_pred])
    loss_layer_2 = Lambda(lambda x: confidence_penalty_weight * confidence_penalty(*x), name="loss_layer_2")([predict, use_pred])
    metric_layer_1 = Lambda(lambda x: recall_at_10(*x), name="metric_layer_1")([target, predict, use_pred])
    
    inputs = [input_0, input_1, input_2, input_3, input_4, target, use_pred]
    outputs = [predict, loss_layer_1, loss_layer_2, metric_layer_1]

    model = keras.models.Model(inputs=inputs, outputs=outputs)

    loss_1 = model.get_layer("loss_layer_1").output
    model.add_loss(loss_1)
    loss_2 = model.get_layer("loss_layer_2").output
    model.add_loss(loss_2)
    metric_1 = model.get_layer("metric_layer_1").output
    model.add_metric(metric_1, name='recall_at_10')

    # Deploying Optimizer.
    model.compile(
        optimizer=Adam(lr=lrate, beta_1=0.9, beta_2=0.999, clipvalue=5.0),
        loss=[None, None, None, None]
    )
    return model