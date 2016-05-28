import pickle
import pylab as plt

D = pickle.load(open("tests/clean.pkl", "r"))
fig, ax = plt.subplots()
fig.canvas.draw()
plt.imshow(D['gauss'].T, cmap='gray_r', origin='lower')
labels = [-5, 5, 15, 25, 35, 45]
ax.set_xticklabels(labels)
ax.get_yticklabels()
ylabels = [-1, 1, 2, 3, 4, 5, 6]
ax.set_yticklabels(ylabels)
ax.set_yticklabels(ylabels, fontsize=45)
ax.set_xticklabels(labels, fontsize=45)
plt.ylabel("m/k", fontsize=50)
plt.savefig("tests/clean_gauss.pdf", transparent=True, bbox_inches='tight',
            pad_inches=0)

fig, ax = plt.subplots()
fig.canvas.draw()
plt.imshow(D['countGauss'].T, cmap='gray_r', origin='lower')
labels = [-5, 5, 15, 25, 35, 45]
ax.set_xticklabels(labels)
ax.get_yticklabels()
# ylabels=[-1, 1, 2, 3, 4, 5, 6]
ylabels = []
ax.set_yticklabels(ylabels)
ax.set_yticklabels(ylabels, fontsize=45)
ax.set_xticklabels(labels, fontsize=45)
# plt.ylabel("m/k", fontsize=50)
plt.savefig("tests/clean_countgauss.pdf", transparent=True, bbox_inches='tight',
            pad_inches=0)
