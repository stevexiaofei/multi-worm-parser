import numpy as np
import matplotlib.pyplot as plt
valid_acc1 = np.load('exp1\\validation_acc.npz')
valid_acc2 = np.load('exp2\\validation_acc.npz')
valid_acc3 = np.load('exp3\\validation_acc.npz')
valid_acc4 = np.load('exp4\\validation_acc.npz')
valid_acc5 = np.load('exp5\\validation_acc.npz')

fig, ax = plt.subplots(figsize=(10,6))
# print(np.max(valid_acc['acc']))
ax.plot(valid_acc1['loss'])
ax.plot(valid_acc2['loss'])
ax.plot(valid_acc3['loss'])
ax.plot(valid_acc4['loss'])
ax.plot(valid_acc5['loss'])
plt.savefig('./tt.jpg',dpi=600)
plt.show()


