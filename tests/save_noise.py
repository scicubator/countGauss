import pickle
import pylab as plt

D = pickle.load(open("tests/noise.pkl", "r"))
fig, ax = plt.subplots()
fig.canvas.draw()
plt.imshow(D['gauss'], cmap='gray_r', origin='lower')
# labels=[-5,0.01, 0.03, 0.08, 0.22,0.6,1]
labels = []
ax.set_xticklabels(labels)
ax.get_yticklabels()
ylabels = [-1, "", 40, ]
ax.set_yticklabels(ylabels)
ax.set_yticklabels(ylabels, fontsize=25)
ax.set_xticklabels(labels, fontsize=25)
plt.ylabel("features", fontsize=22)
plt.savefig("tests/noise_gauss.pdf", transparent=True, bbox_inches='tight',
            pad_inches=0)

fig, ax = plt.subplots()
fig.canvas.draw()
plt.imshow(D['countGauss'], cmap='gray_r', origin='lower')
labels = [-5, 0.01, 0.03, 0.08, 0.22, 0.6, 1]
ax.set_xticklabels(labels)
ax.get_yticklabels()
ylabels = [-1, "", 40, ]
ax.set_yticklabels(ylabels)
ax.set_yticklabels(ylabels, fontsize=25)
ax.set_xticklabels(labels, fontsize=25)
plt.ylabel("sorted", fontsize=22)
plt.savefig("tests/noise_countgauss.pdf", transparent=True, bbox_inches='tight',
            pad_inches=0)
