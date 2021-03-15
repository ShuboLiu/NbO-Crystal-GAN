import numpy as np 
A = [[0, 2.14, 2.14],
     [2.14, 0 ,2.14],
     [2.14, 2.14, 0]
     ]
A = np.array(A)
b = [0.5,0.5,0.5]
b = np.array(b)
p_1 = np.dot(A,b)
print(p_1)
p_2 = [0, 0, 0]
diff = p_1 - p_2
distance = np.linalg.norm(diff, ord=2)
print(distance)