from keras import optimizers


net = [100, -100]
sgd = optimizers.SGD(lr=0.01, clipvalue=1.)
print(net.get_gradients())
