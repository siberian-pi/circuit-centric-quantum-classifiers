"""
test_forward_propagation
*************** circuit ********************
0 ----------  u:  0,pi/4,pi/2,pi/2,pi  -------------------
1 --------------------------------------------|-------------
2--------------------------------------------cp: 1,2,pi/6---
*********************************************
input: [1,2,3,4,5,6,7,8] (unnormalized)
anticipated output:  
"""
import math
import quantum_classifier

pi=math.pi


print("******************* Test1 ****************************")
input_state=[1,2,3,4,5,6,7,8]
quantum_circuit=[['u',[0],[pi/4,pi/2,pi/2,pi]],['cp',[1,2],[pi/6]]]

c=quantum_classifier.Quantum_Classifier(quantum_circuit,3,2)
c.init_state(input_state,3)
c.execute()
print(c.get_state())

print("****************** Test2 ***************************")
input_state=[3,4,6,7,9,12,3,5]
quantum_circuit=[['cp',[2,0],[pi/3]],['cp',[1,2],[pi]],['u',[1],[pi,pi,pi/3,pi/4]]]

d=quantum_classifier.Quantum_Classifier(quantum_circuit,3,3)
d.init_state(input_state,3)
d.execute()
print(d.get_state())
print("**************** anticipated output ************")

