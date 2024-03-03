from customs import nn_constructor as nc
import tensorflow as tf
tfl = tf.keras.layers


def nn_rnn_model(shape, num_of_lstm=2, u_list=None, act='tanh', rec_act='sigmoid', merge_mode='mul', mask=True):
    if u_list is None:
        u_list = [32, 16]
    assert len(u_list) >= num_of_lstm
    # input
    x_in = tf.keras.Input(shape=shape)
    if mask:
        mask_in = x_in[:, :, -1:]
        x = x_in[:, :, :-1]
        mask_lstm = tf.cast(mask_in[:, :, 0],bool)  # используем только то число шагов по времени, что есть в событии
    else:
        x = x_in[:, :, :]
        mask_lstm = None
    ######
    # Добавляем LSTM слой с return_sequences = True.
    ######

    # first LSTMs
    for i in range(int(num_of_lstm) - 1):
        u = u_list[i]
        x = nc.BidirClass(u, True, act=act, rec_act=rec_act, merge_mode=merge_mode).block(x, mask_lstm)
        # не умножаем на маску, далее она прописана
    ######
    # Добавляем LSTM слой с return_sequences = False, так как НЕ хотим зависеть от input shape.
    ######
    # last LSTM
    u = u_list[int(num_of_lstm) - 1]
    bidir2 = nc.BidirClass(u, False, act=act, rec_act=rec_act, merge_mode=merge_mode)
    x = bidir2.block(x, mask_lstm)
    # + Dense
    # softmax
    outputs = tfl.Dense(2, activation='softmax')(x)
    model = tf.keras.Model(inputs=x_in, outputs=outputs)
    return model


def nn_main_model_NoMaskAsChannel(shape, u1=32, u2=8, f_id_list=None, k_id_list=None,
                                  f_cd_list=None, k_cd_list=None, s_cd_list=None):
    # input
    if f_id_list is None:
        f_id_list = [64, 128, 64]
    if k_id_list is None:
        k_id_list = [12, 14, 4]
    if f_cd_list is None:
        f_cd_list = [64, 128, 64]
    if k_cd_list is None:
        k_cd_list = [12, 14, 4]
    if s_cd_list is None:
        s_cd_list = [2, 2, 2]
    x_input = tf.keras.Input(shape)
    mask_in = x_input[:, :, -1:]
    x_in = x_input[:, :, :-1]
    print(x_in.get_shape())
    ######
    # Добавляем LSTM слой с return_sequences = True.
    ######
    # 1 LSTM
    bidir1 = nc.BidirClass(u1, True, act='tanh', rec_act='sigmoid', merge_mode='mul')
    mask_lstm = tf.cast(mask_in[:, :, 0], bool)
    x = bidir1.block(x_in, mask_lstm) * mask_in
    ######
    # 2
    # Блок resnet
    x, mask = nc.res_block(x, mask_in, f_id=f_id_list[0], k_id=k_id_list[0],
                           f_cd=f_cd_list[0], k_cd=k_cd_list[0], s_cd=s_cd_list[0])
    ######
    # 3
    x, mask = nc.res_block(x, mask, f_id=f_id_list[1], k_id=k_id_list[1],
                           f_cd=f_cd_list[1], k_cd=k_cd_list[1], s_cd=s_cd_list[1])
    ######
    # 4
    x, mask = nc.res_block(x, mask, f_id=f_id_list[2], k_id=k_id_list[2],
                           f_cd=f_cd_list[2], k_cd=k_cd_list[2], s_cd=s_cd_list[2])
    ######
    # Добавляем LSTM слой с return_sequences = False, так как НЕ хотим зависеть от input shape.
    ######
    # 5 LSTM
    bidir2 = nc.BidirClass(u2, False, act='tanh', rec_act='sigmoid', merge_mode='mul')
    mask_lstm = tf.cast(mask[:, :, 0], bool)
    x = bidir2.block(x, mask_lstm)
    # + Dense
    # softmax
    outputs = tfl.Dense(2, activation='softmax')(x)
    model = tf.keras.Model(inputs=x_input, outputs=outputs)
    return model
