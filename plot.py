import numpy as np
import matplotlib.pyplot as plt
valid_acc = np.load('exp1\\validation_acc.npz')
fig, ax = plt.subplots(figsize=(10,6))
print(np.max(valid_acc['acc']))
ax.plot(valid_acc['acc'])
plt.savefig('./tt.jpg',dpi=600)
plt.show()


