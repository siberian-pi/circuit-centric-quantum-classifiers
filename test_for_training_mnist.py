#Test for function update
import quantum_classifier
#初始化,qubit_number,circuit
classifier=quantum_classifier.Quantum_Classifier()
train_file='mnist/train-images.idx3-ubyte'
label_file='mnist/train-labels.idx1-ubyte'
classifier.train_mnist(train_file,label_file)



import pickle
fp=open('model20180803.pickle','wb')
pickle.dump(classifier,fp)
fp.close()

"""
fp=open('model_pickle.txt','rb')
circuit=pickle.load(fp)
fp.close()
"""
