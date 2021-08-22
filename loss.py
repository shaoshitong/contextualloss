import tensorflow as tf
def LossFunction(learning_rate=0.01):
    def loss(x,y):
        return tf.sigmoid(-tf.abs(tf.subtract(x,y)))*learning_rate
    return loss
