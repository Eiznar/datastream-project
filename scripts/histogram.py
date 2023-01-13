import matplotlib.animation as animation
import datetime as dt
import matplotlib.pyplot as plt

real_predictions_proba = []

fig = plt.figure(figsize=(8,14))
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

barh_labels = ["fake", "real"]

with open('archive.csv', encoding="utf8") as f:
    y = [[x for x in line.split(';')][2] for line in f] # {'tweet': tweet,'fake': fake, 'real': real, "prediction": pred}
    
with open('archive.csv', encoding="utf8") as f:
    pred = [[x for x in line.split(';')][3] for line in f] # {'tweet': tweet,'fake': fake, 'real': real, "prediction": pred}
    

def animate(i):
    # First plot /// Store every prediction iteratively
    if i < len(y) :
        real_predictions_proba.append(float("{:.2f}".format(float(y[i]))))
    else :
        i = len(y)-1

    # Clear plot for iterative processing and plot
    ax1.clear()
    ax1.hist(real_predictions_proba, bins = len(y), density= True)

    # Set ax1 axis labels
    ax1.set_title('Distribution of "real" prediction of the model')
    ax1.set_xlabel('Probability predicted')
    ax1.set_ylabel('Number of elements')

    # Second plot
    # Clear plot for iterative processing and plot
    ax2.clear()
    predictions_count =  [pred[:i].count('fake\n'),pred[:i].count('real\n')]
    rects = ax2.barh(barh_labels, predictions_count)

    # Set ax2 axis labels
    ax2.bar_label(rects, predictions_count,
                  padding=5, color='black', fontweight='bold')
    ax2.set_title('Bar plot of predictions class of the model')
    ax2.set_xlabel('Number of elements predicted')
    ax2.set_ylabel('Prediction class')

    fig.tight_layout()

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()
