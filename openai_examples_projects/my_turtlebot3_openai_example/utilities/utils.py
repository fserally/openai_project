import visdom
import numpy as np
from gym import wrappers
import rospkg
import itertools

def expand(lst, n):
    lst = [[i]*n for i in lst]
    lst = list(itertools.chain.from_iterable(lst))
    return lst

vis = visdom.Visdom()
win = "win"
# line updates

win = vis.line(
    X=np.column_stack(([0], [0])),
    Y=np.column_stack(([0],
                       [0])),
    win=win,
)
rospack = rospkg.RosPack()
pkg_path = rospack.get_path('my_turtlebot3_openai_example')
outdir = pkg_path + '/training_results'
results = wrappers.monitor.load_results(outdir)
data = results['episode_rewards']
labels = []
for i in range(int(len(data))): labels.append(i)


avg_data = []

#average
average = 50
for i, val in enumerate(data):
    if i%average==0:
        if (i+average) < len(data)+average:
            avg =  sum(data[i:i+average])/average
            avg_data.append(avg)
average = expand(avg_data,average)
average = average[:len(data)]

average_labels = []
for i in range(int(len(average))): average_labels.append(i)

print(data)
print(average)



vis.line(
    X=np.column_stack((labels, average_labels)),
    Y=np.column_stack((data, average)),
    win=win,
    update='append',
    opts=dict(
        title="Training Results",
        legend=["results","Average"],
    )
)







