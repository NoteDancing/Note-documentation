import tensorflow as tf
import Note.nn.layer.dense as d
from Note.nn.layer.flatten import flatten


class nn:
    def __init__(self):
        self.loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.opt=tf.keras.optimizers.Adam()
    
    
    def build(self):
        self.layer1=d.dense(128,784,activation='relu')
        self.layer2=d.dense(10,128)
        self.param=[self.layer1.param,self.layer2.param]
        return
    
    
    def fp(self,data):
        data=flatten(data)
        output1=self.layer1.output(data)
        output2=self.layer2.output(output1)
        return output2
    
    
    def loss(self,output,labels):
        return self.loss_object(labels,output)