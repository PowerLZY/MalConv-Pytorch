import matplotlib.pyplot as plt
import numpy as np


# numpy读取

a=np.load('tr_acc.npy')
a=a.tolist()

plt.figure()
plt.plot(range(len(a)),a , c='r', label='Training Set', linewidth=2)
#plt.plot(range(len(Test_TPR)), Test_TPR, c='g', linestyle='--', label='Validation Set', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('TPR')
plt.legend()
#plt.savefig('saves/Epoch_TPR({0}, {1}).png'.format(self.blackbox, flag[self.same_train_data]))
plt.show()