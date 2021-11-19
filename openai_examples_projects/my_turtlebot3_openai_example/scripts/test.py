'''import matplotlib.pyplot as plt

# Define x and y values
x = [7, 14, 21, 28, 35, 42, 49]
y = [8, 13, 21, 30, 31, 44, 50]

# Plot a simple line chart without any feature
plt.plot(x, y)
plt.show()'''
'''
import visdom
from time import sleep

import numpy as np
vis = visdom.Visdom()

'''
'''
trace = dict(x=[], y=[], mode="markers+lines", type='custom',
             marker={'color': 'red', 'symbol': 104, 'size': "10"},
             text=["one", "two", "three"], name='1st Trace')

average_trace = dict(x=[], y=[], mode="markers+lines", type='custom',
             marker={'color': 'blue', 'symbol': 104, 'size': "10"},
             text=["one", "two", "three"], name='Average Trace')

layout = dict(title="Training Results", xaxis={'title': 'Episodes'}, yaxis={'title': 'Total Rewards'})
layout2 = dict(title="Training Results", xaxis={'title': 'Episodes'}, yaxis={'title': 'Total Rewards'})

trace['x'] = np.column_stack(([10,20,30],[1,2,3]))
trace['y'] = np.column_stack(([40, 50, 60],[4, 5, 6]))

vis._send({'data': [trace], 'layout': layout, 'win': 'mywin'})
sleep(5)

'''
'''
viz = visdom.Visdom()

# line updates
win = viz.line(
    X=np.column_stack((np.arange(0, 10), np.arange(0, 10))),
    Y=np.column_stack((np.linspace(5, 10, 10),
                       np.linspace(5, 10, 10) + 5)),
)

viz.line(
    X=np.column_stack((np.arange(50, 60), np.arange(50, 60))),
    Y=np.column_stack((np.linspace(5, 10, 10),
                       np.linspace(5, 10, 10) + 5)),
)

# Save or pass win to the other scripts and use it to append to this window

# In other scripts:
'''
'''
viz.line(
    X=np.column_stack((np.arange(50, 60), np.arange(50, 60))),
    Y=np.column_stack((np.linspace(5, 10, 10),
                       np.linspace(5, 10, 10) + 5)),
    win=win,
    update='append'
)

'''

'''
from utilities import utils

global plotter

plotter = utils.VisdomLinePlotter(env_name="Turtlebot Plots")

x = 7
y = 8

plotter.plot('loss', 'train', 'Class Loss', x, y)
plotter.plot('acc', 'val', 'Class Accuracy', x, y)

import visdom
import numpy as np
vis = visdom.Visdom()

trace = dict(x=[], y=[], mode="markers+lines", type='custom',
             marker={'color': 'red', 'symbol': 104, 'size': "10"},
             text=["one", "two", "three"], name='1st Trace')

layout = dict(title="Training Results", xaxis={'title': 'Episodes'}, yaxis={'title': 'Total Rewards'})

x = [7, 14, 21, 28, 35, 42, 49]
y = [8, 13, 21, 30, 31, 44, 50]

trace['x'] = x
trace['y'] = y

vis._send({'data': [trace], 'layout': layout, 'win': 'testmywin'})

win = "win"
# line updates
win = vis.line(
    X=np.column_stack(([7], [14])),
    Y=np.column_stack(([45],
                       [15])),
    win=win,
)


vis.line(
    X=np.column_stack(([7,14], [14,21])),
    Y=np.column_stack(([45,40],
                       [15,49])),
    win=win,
    update='append',
    opts=dict(
        title="Training Results",
        legend=["results","Average"],
    )
)

'''
import pickle
import rospkg
import rospy
import numpy as np

import pickle
import visdom
import itertools
import rospkg

rospack = rospkg.RosPack()
pkg_path = rospack.get_path('my_turtlebot3_openai_example')

run_mode = rospy.get_param("/turtlebot3/nsteps")
print(run_mode)

q_table = dict(pickle.load(open(pkg_path + '/models/model_20211110_16:19:2912/turtlebot3_100model.pkl', 'rb')))
actions = [0,1,2]


q = [q_table.get(('00001', a), 0.0) for a in actions]
maxQ = max(q)
print(maxQ)
