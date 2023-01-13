import matplotlib.animation as animation
import datetime as dt
import matplotlib.pyplot as plt
import random


with open('archive.csv', encoding="utf8") as f:
    y = [[x for x in line.split(';')][1] for line in f]

real_predictions_proba = []

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
xs = []
ys = []

def animate(i, xs, ys):
    # Add x and y to lists
    xs.append(dt.datetime.now().strftime('%H:%M:%S.%f'))
    real_predictions_proba.append(float("{:.2f}".format(float(y[i]))))


    # Limit x and y lists to 20 items
    xs = xs[-20:]
    ys = real_predictions_proba[-20:]

    # Draw x and y lists
    ax.clear()
    ax.plot(xs, ys)

    # Format plot
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title('Distribution')
    plt.ylabel('Probability')

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(fig, animate, fargs=(xs, ys), interval=2000)
plt.show()
