class bidir_class:
    def __init__(self, u, return_sequences = False, act = 'tanh', rec_act = 'sigmoid', merge_mode = 'mul'):
        self.u = u 
        self.rs = return_sequences
        self.act = act
        self.rec_act = rec_act
        self.mm = merge_mode
        self.tfl = tf.keras.layers
        
    def layer(self):
        lstm_layer = tfl.LSTM(self.u, activation=self.act, recurrent_activation=self.rec_act, return_sequences=self.rs)
        return tfl.Bidirectional(lstm_layer, merge_mode=self.mm)
    
    def block(self, x, mask_lstm = None):
        if mask_lstm!=None:
            #print(mask_lstm.shape,x.shape)z
            x = self.layer()(x, mask = mask_lstm)
        else:
            x = self.layer()(x)
        x = tfl.BatchNormalization(axis=-1)(x)
        return x

#LSTM = рекуррентные слои
#RB = Res-Net Block
#Name shows the sequence of layers
def nn_1LSTM_3RB_1LSTM_tuned1(Shape, u1 = 8, u2 = 16, f_id_list = [32,32,32], k_id_list = [8,16,20],
                              f_cd_list = [32,32,32], k_cd_list = [8,16,20], s_cd_list = [2,2,2]):
    # input
    x_in = tf.keras.Input(shape=Shape)
    mask_in = x_in[:,:,-1:]
    ######
    #Добавляем LSTM слой с return_sequences = True.
    ######
    #1 LSTM
    bidir1 = bidir_class(u1, True, act = 'tanh', rec_act = 'sigmoid', merge_mode = 'mul')
    mask_lstm = tf.cast(mask_in[:,:,0],bool)
    x = bidir1.block(x_in, mask_lstm)*mask_in
    ######
    #2
    #Блок resnet
    x, mask = res_block(x, mask_in, f_id = f_id_list[0], k_id = k_id_list[0],
                        f_cd = f_cd_list[0], k_cd = k_cd_list[0], s_cd = s_cd_list[0])
    ######
    #3
    x, mask = res_block(x, mask, f_id = f_id_list[1], k_id = k_id_list[1],
                        f_cd = f_cd_list[1], k_cd = k_cd_list[1], s_cd = s_cd_list[1])
    ######
    #4
    x, mask = res_block(x, mask, f_id = f_id_list[2], k_id = k_id_list[2],
                        f_cd = f_cd_list[2], k_cd = k_cd_list[2], s_cd = s_cd_list[2])
    ######
    #Добавляем LSTM слой с return_sequences = False, так как НЕ хотим зависеть от input shape.
    ######
    #5 LSTM
    bidir2 = bidir_class(u2, False, act = 'tanh', rec_act = 'sigmoid', merge_mode = 'mul')
    mask_lstm = tf.cast(mask[:,:,0],bool)
    x = bidir2.block(x, mask_lstm)
    #+ Dense
    #softmax
    outputs = tfl.Dense(2,activation = 'softmax')(x) 
    model = tf.keras.Model(inputs=x_in, outputs=outputs) 
    return model

def nn_main_model(Shape, u1 = 8, u2 = 16, f_id_list = [64,128,64], k_id_list = [12,14,4],
                              f_cd_list = [64,128,64], k_cd_list = [12,14,4], s_cd_list = [2,2,2]):
    # input
    x_in = tf.keras.Input(shape=Shape)
    mask_in = x_in[:,:,-1:]
    ######
    #Добавляем LSTM слой с return_sequences = True.
    ######
    #1 LSTM
    bidir1 = bidir_class(u1, True, act = 'tanh', rec_act = 'sigmoid', merge_mode = 'mul')
    mask_lstm = tf.cast(mask_in[:,:,0],bool)
    x = bidir1.block(x_in, mask_lstm)*mask_in
    ######
    #2
    #Блок resnet
    x, mask = res_block(x, mask_in, f_id = f_id_list[0], k_id = k_id_list[0],
                        f_cd = f_cd_list[0], k_cd = k_cd_list[0], s_cd = s_cd_list[0])
    ######
    #3
    x, mask = res_block(x, mask, f_id = f_id_list[1], k_id = k_id_list[1],
                        f_cd = f_cd_list[1], k_cd = k_cd_list[1], s_cd = s_cd_list[1])
    ######
    #4
    x, mask = res_block(x, mask, f_id = f_id_list[2], k_id = k_id_list[2],
                        f_cd = f_cd_list[2], k_cd = k_cd_list[2], s_cd = s_cd_list[2])
    ######
    #Добавляем LSTM слой с return_sequences = False, так как НЕ хотим зависеть от input shape.
    ######
    #5 LSTM
    bidir2 = bidir_class(u2, False, act = 'tanh', rec_act = 'sigmoid', merge_mode = 'mul')
    mask_lstm = tf.cast(mask[:,:,0],bool)
    x = bidir2.block(x, mask_lstm)
    #+ Dense
    #softmax
    outputs = tfl.Dense(2,activation = 'softmax')(x)
    model = tf.keras.Model(inputs=x_in, outputs=outputs)
    return model

def nn_main_model_NoMaskAsChannel(Shape, u1 = 8, u2 = 16, f_id_list = [64,128,64], k_id_list = [12,14,4],
                              f_cd_list = [64,128,64], k_cd_list = [12,14,4], s_cd_list = [2,2,2]):
    # input
    x_input = tf.keras.Input(Shape)
    mask_in = x_input[:,:,-1:]
    x_in = x_input[:,:,:-1]
    print(x_in.get_shape())
    ######
    #Добавляем LSTM слой с return_sequences = True.
    ######
    #1 LSTM
    bidir1 = bidir_class(u1, True, act = 'tanh', rec_act = 'sigmoid', merge_mode = 'mul')
    mask_lstm = tf.cast(mask_in[:,:,0],bool)
    x = bidir1.block(x_in, mask_lstm)*mask_in
    ######
    #2
    #Блок resnet
    x, mask = res_block(x, mask_in, f_id = f_id_list[0], k_id = k_id_list[0],
                        f_cd = f_cd_list[0], k_cd = k_cd_list[0], s_cd = s_cd_list[0])
    ######
    #3
    x, mask = res_block(x, mask, f_id = f_id_list[1], k_id = k_id_list[1],
                        f_cd = f_cd_list[1], k_cd = k_cd_list[1], s_cd = s_cd_list[1])
    ######
    #4
    x, mask = res_block(x, mask, f_id = f_id_list[2], k_id = k_id_list[2],
                        f_cd = f_cd_list[2], k_cd = k_cd_list[2], s_cd = s_cd_list[2])
    ######
    #Добавляем LSTM слой с return_sequences = False, так как НЕ хотим зависеть от input shape.
    ######
    #5 LSTM
    bidir2 = bidir_class(u2, False, act = 'tanh', rec_act = 'sigmoid', merge_mode = 'mul')
    mask_lstm = tf.cast(mask[:,:,0],bool)
    x = bidir2.block(x, mask_lstm)
    #+ Dense
    #softmax
    outputs = tfl.Dense(2,activation = 'softmax')(x)
    model = tf.keras.Model(inputs=x_input, outputs=outputs)
    return model

def nn_rnn_model(Shape, num_of_lstm = 2, u_list = [32,16], act = 'tanh', rec_act = 'sigmoid', merge_mode = 'mul', mask = True):
    assert len(u_list)>= num_of_lstm
    # input
    x_in = tf.keras.Input(shape=Shape)
    if mask:
        mask_in = x_in[:,:,-1:]
        mask_lstm = tf.cast(mask_in[:, :, 0], bool)  # используем только то число шагов по времени, которое есть в событии
    else:
        mask_lstm = None
    ######
    #Добавляем LSTM слой с return_sequences = True.
    ######
    
    #first LSTMs
    for i in range(int(num_of_lstm)-1):
        u = u_list[i]
        bidir1 = bidir_class(u, True, act = act, rec_act = rec_act, merge_mode = merge_mode)
        x = bidir1.block(x_in, mask_lstm) #не умножаем на маску, далее она прописана
    ######
    #Добавляем LSTM слой с return_sequences = False, так как НЕ хотим зависеть от input shape.
    ######
    #last LSTM
    u = u_list[int(num_of_lstm)-1]
    bidir2 = bidir_class(u, False, act = act, rec_act = rec_act, merge_mode = merge_mode)
    x = bidir2.block(x, mask_lstm)
    #+ Dense
    #softmax
    outputs = tfl.Dense(2,activation = 'softmax')(x)
    model = tf.keras.Model(inputs=x_in, outputs=outputs)
    return model