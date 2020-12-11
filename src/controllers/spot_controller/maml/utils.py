import matplotlib.pyplot as plt


def plot_rewards(rew):
    reww = []
    for i in rew:
        reww.append(i.detach().cpu())
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = [i for i in range(len(reww))]
    ax.plot(reww)
    plt.show()
