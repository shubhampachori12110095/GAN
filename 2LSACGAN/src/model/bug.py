import numpy as np
import keras

def make_trainable(net, value):
    for l in net.layers:
        l.trainable = value

####GEN model
gen_input = keras.layers.Input(shape=(3,))
y = keras.layers.Dense(5)(gen_input)
GEN = keras.models.Model(input=gen_input, output=y,name="GEN")
GEN.trainable = True
GEN.compile(optimizer='rmsprop', loss='mse')

###DISC model
disc_input =  keras.layers.Input(shape=(5,))
z = keras.layers.Dense(5)(disc_input)
DISC = keras.models.Model(input=disc_input, output=z,name="DISC")
DISC.compile(optimizer='rmsprop', loss='mse')


##DCGAN model
gen_input = keras.layers.Input(shape=(3,))
gen_output = GEN([gen_input])
disc_output = DISC([gen_output])
disc_output = keras.layers.Dense(5)(disc_output)
DCGAN = keras.models.Model(input=gen_input,output=disc_output,name="DCGAN")

make_trainable(GEN,False)
DCGAN.compile(optimizer='rmsprop', loss='mse')


###TRAINING GEN:
data_x = np.random.random((4, 3))
data_y = np.random.random((4, 5))
print ("\n training GEN:")
GEN.train_on_batch(data_x, data_y)
out1=GEN.predict(data_x)
print out1


#TRAINING DISC:
#data_z = np.ones((4, 3))
#gen_out=GEN.predict(data_z)
#data_w = np.ones((4, 5))
#print ("\n training DISC:")
#DISC.fit(gen_out, data_w, nb_epoch=2)
#out2=DISC.predict(gen_out)
#print out2

##TRAINING DCGAN:
data_a = np.ones((4, 3)) *2
data_b = np.ones((4, 5)) *3
print ("\n training GAN:")
DCGAN.train_on_batch(data_a,data_b)
DCGAN.train_on_batch(data_a,data_b)
DCGAN.train_on_batch(data_a,data_b)
dcgan_out=DCGAN.predict(data_a)

out1=GEN.predict(data_x)
print out1




