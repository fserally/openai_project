#!/usr/bin/env python
import json
import gym
import numpy
import time
import os
import qlearn
import visdom
from gym import wrappers
# ROS packages required
import rospy
import numpy as np

import pickle
import visdom
import itertools
import rospkg
import os
from datetime import datetime

from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment

def createModelDirectory():
    pkg_path = rospack.get_path('my_turtlebot3_openai_example')
    today = datetime.now()
    if today.hour < 12:
        h = "00"
    else:
        h = "12"

    folder_name = 'model_' + today.strftime('%Y%m%d_%H:%M:%S')+ h
    folder_path = pkg_path + '/models/' + folder_name
    os.mkdir(folder_path)
    return folder_path

def expand(lst, n):
    lst = [[i]*n for i in lst]
    lst = list(itertools.chain.from_iterable(lst))
    return lst

if __name__ == '__main__':

    ###### Fatimah 25/10/2021 #######
    vis = visdom.Visdom() #for visualization in real time
    win = "win"
    win = vis.line(
        X=np.column_stack(([0], [0])),
        Y=np.column_stack(([0],
                           [0])),
        win=win,
    )

    #initialize layout of line graph
    trace = dict(x=[], y=[], mode="markers+lines", type='custom',
             marker={'color': 'red', 'symbol': 104, 'size': "10"},
             text=["one", "two", "three"], name='1st Trace')

    trace_loss = dict(x=[], y=[], mode="markers+lines", type='custom',
             marker={'color': 'red', 'symbol': 104, 'size': "10"},
             text=["one", "two", "three"], name='2nd Trace')
    layout = dict(title="Training Results", xaxis={'title': 'Episodes'}, yaxis={'title': 'Total Rewards'})

    layout_loss = dict(title="Decreasing probability of choosing a random action", xaxis={'title': 'Episodes'}, yaxis={'title': 'Epsilon'})
    ###### Fatimah 25/10/2021 #######

    rospy.init_node('turtlebot3_world_qlearn', anonymous=True, log_level=rospy.DEBUG)

    # Init OpenAI_ROS ENV
    task_and_robot_environment_name = rospy.get_param(
        '/turtlebot3/task_and_robot_environment_name')
    env = StartOpenAI_ROS_Environment(
        task_and_robot_environment_name)
    # Create the Gym environment

    rospy.loginfo("Gym environment done")
    rospy.loginfo("Starting Learning")

    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('my_turtlebot3_openai_example')
    outdir = pkg_path + '/training_results'
    outdir_model = createModelDirectory()

    env = wrappers.Monitor(env, outdir, force=True)
    #env.spec.timestep_limit = 1000

    rospy.loginfo("Monitor Wrapper started")

    last_time_steps = numpy.ndarray(0)

    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file
    Alpha = rospy.get_param("/turtlebot3/alpha")
    Epsilon = rospy.get_param("/turtlebot3/epsilon")
    Gamma = rospy.get_param("/turtlebot3/gamma")
    epsilon_discount = rospy.get_param("/turtlebot3/epsilon_discount")
    nepisodes = rospy.get_param("/turtlebot3/nepisodes")
    nsteps = rospy.get_param("/turtlebot3/nsteps")
    running_step = rospy.get_param("/turtlebot3/running_step")
    run_mode = rospy.get_param("/turtlebot3/run_mode")
    timestep_limit = rospy.get_param("/turtlebot3/timestep_limit")

    env._max_episode_steps = nsteps
    env.spec.timestep_limit = timestep_limit

    # Initialises the algorithm that we are going to use for learning
    qlearn = qlearn.QLearn(actions=range(env.action_space.n),
                               alpha=Alpha, gamma=Gamma, epsilon=Epsilon)

    if run_mode == "evaluation":
        evaluation_model_path = rospy.get_param("/turtlebot3/evaluation_model_path")
        q_table = dict(pickle.load(open(pkg_path + '/models/' + evaluation_model_path, 'rb')))
        qlearn.setQ(q_table)
        qlearn.setEpsilon(0)

    initial_epsilon = qlearn.epsilon

    start_time = time.time()
    highest_reward = 0
    episodes = []
    rewards = []
    epsilon = []
    done = False
    # Starts the main training loop: the one about the episodes to do
    for x in range(nepisodes):
        done = False
        rospy.logdebug("############### START EPISODE=>" + str(x))

        cumulated_reward = 0
        # Initialize the environment and get first state of the robot
        observation = env.reset()

        if qlearn.epsilon > 0.05:
            qlearn.epsilon *= epsilon_discount

        state = ''.join(map(str, observation))


        # Show on screen the actual situation of the robot
        # env.render()
        # for each episode, we test the robot until it crashes
        i = 0
        #for i in range(nsteps):
        while True:
            i = i + 1
            rospy.logwarn("############### Start Step=>" + str(i))
            # Pick an action based on the current state
            action = qlearn.chooseAction(state)

            rospy.logdebug("Next action is:%d", action)
            # Execute the action in the environment and get feedback
            observation, reward, done, info = env.step(action)

            rospy.logdebug(str(observation) + " " + str(reward))
            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            nextState = ''.join(map(str, observation))

            # Make the algorithm learn based on the results
            rospy.logdebug("# state we were=>" + str(state))
            rospy.logdebug("# action that we took=>" + str(action))
            rospy.logdebug("# reward that action gave=>" + str(reward))
            rospy.logdebug("# episode cumulated_reward=>" + str(cumulated_reward))
            rospy.logdebug("# State in which we will start next step=>" + str(nextState))
            qlearn.learn(state, action, reward, nextState) #to check
            env._flush(force=True)
            if not (done):
                rospy.logdebug("NOT DONE")
                state = nextState
            else:
                rospy.logdebug("DONE")
                last_time_steps = numpy.append(last_time_steps, [int(i + 1)])
                break
            rospy.logwarn("############### END Step=>" + str(i))
            #raw_input("Next Step...PRESS KEY")
            # rospy.sleep(2.0)

        episodes.append(x+1)
        rewards.append(cumulated_reward)
        epsilon.append(round(qlearn.epsilon, 2))

        ###### plot graph #####
        trace['x'] = episodes
        trace['y'] = rewards
        labels = []
        for i in range(int(len(rewards))): labels.append(i)
        avg_data = []
        #average
        average = 50
        for i, val in enumerate(rewards):
            if i%average==0:
                if (i+average) < len(rewards)+average:
                    avg =  sum(rewards[i:i+average])/average
                    avg_data.append(avg)
        average = expand(avg_data,average)
        average = average[:len(rewards)]

        average_labels = []
        for i in range(int(len(average))): average_labels.append(i)



        trace_loss['x'] = episodes
        trace_loss['y'] = epsilon

        vis.line(
            X=np.column_stack((labels, average_labels)),
            Y=np.column_stack((rewards, average)),
            win=win,
            opts=dict(
                title="Training Results" if run_mode == "training" else "Evaluation Results",
                legend=["results","Average"],
            )
        )

        vis._send({'data': [trace], 'layout': layout, 'win': 'mywin'})
        vis._send({'data': [trace_loss], 'layout': layout_loss, 'win': 'winloss'})

        if run_mode == "training":
            if (x+1) % 100 == 0 and x > 0:
                model_name = "turtlebot3_" + str(x+1)
                model_name_path = os.path.join(outdir_model,model_name)
                pickle.dump(qlearn.getQTable(),open(model_name_path + 'model.pkl', 'wb'))


        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        rospy.logerr(("EP: " + str(x + 1) + " - [alpha: " + str(round(qlearn.alpha, 2)) + " - gamma: " + str(
            round(qlearn.gamma, 2)) + " - epsilon: " + str(round(qlearn.epsilon, 2)) + "] - Reward: " + str(
            cumulated_reward) + "     Time: %d:%02d:%02d" % (h, m, s)))


    rospy.loginfo(("\n|" + str(nepisodes) + "|" + str(qlearn.alpha) + "|" + str(qlearn.gamma) + "|" + str(
        initial_epsilon) + "*" + str(epsilon_discount) + "|" + str(highest_reward) + "| PICTURE |"))

    l = last_time_steps.tolist()
    l.sort()

    # print("Parameters: a="+str)
    rospy.loginfo("Overall score: {:0.2f}".format(last_time_steps.mean()))

    #rospy.loginfo("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    env.close()

