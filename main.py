import math
import numpy as np
import random

#Custom created Functions of Neurons
class nnf():
    def sigmoid(self,x):
        return (1/(1+(math.e**-x)))
    
    def relu(self,x):
        if x<0:
            return 0
        else:
            return 1
    
    def err_rate(self,x):
        return (x**2)

    def gradient_descent(self,x,err,learn_rate):
        return x*learn_rate*err

    def rand_weights(self):
        w = np.array([random.random(),random.random(),random.random(),random.random()])
        random.seed(1)
        for _ in range(3):
            w = np.vstack((w,np.array([[random.random(),random.random(),random.random(),random.random()]])))
        return w


#Artifical Neuron Inherting Functions
class ann(nnf):
    def inputs(self,train):
        self.train = train
        self.weights = nnf.rand_weights(self)

    #*Important Part
    def feedforward(self,bias):
        #Perform w*x + b
        print(self.weights)
        temp = np.matmul(self.weights,self.train) + bias

        #Perform Sigmoid/ReLU Activation Function
        for i in range(4):
            for j in range(4):
                temp[i,j] = nnf.sigmoid(self,temp[i,j])

        #Error Function -> Mean Squared Error
        print(temp)
        print('Calculating Error')
        err = np.subtract(self.train,temp)
        print(err)
        for i in range(4):
            for j in range(4):
                err[i,j] = nnf.err_rate(self,err[i,j])
        err = err/2

        #Implementation of gradient descent
        del_w = temp
        for i in range(4):
            for j in range(4):
                del_w[i,j] = nnf.gradient_descent(self,temp[i,j],err[i,j],0.01)
        
        #Backpropogation
        self.weights = np.add(self.weights,del_w)

    def run(self):
        print(nnf.log_err(self,1.001))


if __name__ == "__main__":
    test_net = ann()
    test_net.inputs(np.array([[1,0,1,0],[1,1,0,1],[1,0,1,1],[0,1,0,1]]))
    for i in range(100):
        print('Epoch : %d' % (i+1))
        test_net.feedforward(5)