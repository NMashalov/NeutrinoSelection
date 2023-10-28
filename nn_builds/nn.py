# LSTM = рекуррентные слои
import tensorflow as tf

tfl = tf.keras.layers


def nn_rnn_model(Shape, num_of_lstm=2, u_list=None, act='tanh', rec_act='sigmoid', merge_mode='mul', mask=True):
    if u_list is None:
        u_list = [32, 16]
    assert len(u_list) >= num_of_lstm
    # input
    x_in = tf.keras.Input(shape=Shape)
    if mask:
        mask_in = x_in[:, :, -1:]
        x = x_in[:, :, :-1]
        mask_lstm = tf.cast(mask_in[:, :, 0],
                            bool)  # используем только то число шагов по времени, которое есть в событии
    else:
        x = x_in[:, :, :]
        mask_lstm = None
    ######
    # Добавляем LSTM слой с return_sequences = True.
    ######

    # first LSTMs
    for i in range(int(num_of_lstm) - 1):
        u = u_list[i]
        x = nc.bidir_class(u, True, act=act, rec_act=rec_act, merge_mode=merge_mode).block(x, mask_lstm)
        # не умножаем на маску, далее она прописана
    ######
    # Добавляем LSTM слой с return_sequences = False, так как НЕ хотим зависеть от input shape.
    ######
    # last LSTM
    u = u_list[int(num_of_lstm) - 1]
    bidir2 = nc.bidir_class(u, False, act=act, rec_act=rec_act, merge_mode=merge_mode)
    x = bidir2.block(x, mask_lstm)
    # + Dense
    # softmax
    outputs = tfl.Dense(2, activation='softmax')(x)
    model = tf.keras.Model(inputs=x_in, outputs=outputs)
    return model
