import matplotlib.pyplot as plt


def plot_rewards(rew):
    reww = []
    for i in rew:
        reww.append(i)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(reww)
    plt.show()


def plot_gradient_norms(grads):
    reww = []
    for i in grads:
        reww.append(i.cpu())
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(reww)
    plt.show()
