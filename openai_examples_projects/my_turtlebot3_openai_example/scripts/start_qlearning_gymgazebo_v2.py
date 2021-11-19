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
import h5py
import rospkg
import matplotlib.pyplot as plt

import random
import math
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
#from keras.models import model_from_yaml
import pickle
from keras.models import model_from_json
import visdom

import rospkg
import os


from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment

def save(self, model_name, qtable, models_dir_path="/tmp"):
        """
        We save the current model
        """
        model_name_json_format = model_name+".json"
        model_name_json_format_path = os.path.join(models_dir_path,model_name_json_format)

        json_object = json.dumps(qtable, indent = 4)

        with open(model_name_json_format_path, "w") as json_file:
            json_file.write(json_object)
        print("Saved model to disk")

if __name__ == '__main__':

    ###### Fatimah 25/10/2021 #######
    vis = visdom.Visdom() #for visualization in real time
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
    env = wrappers.Monitor(env, outdir, force=True)
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
    run_mode = rospy.get_param("/mode")

    env._max_episode_steps = nsteps

    if run_mode == 'training':
        # Initialises the algorithm that we are going to use for learning
        qlearn = qlearn.QLearn(actions=range(env.action_space.n),
                               alpha=Alpha, gamma=Gamma, epsilon=Epsilon, )

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
        # for each episode, we test the robot for nsteps

        for i in range(nsteps):

            rospy.logwarn("############### Start Step=>" + str(i))
            # Pick an action based on the current state
            if run_mode == 'evaluation':
                q_table = dict(pickle.load(open('models/turtlebot3_2800.jsonmodel.pkl', 'rb')))
                action = np.argmax(q_table[state])
            else:
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
            qlearn.learn(state, action, reward, nextState)
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
        trace_loss['x'] = episodes
        trace_loss['y'] = epsilon

        vis._send({'data': [trace], 'layout': layout, 'win': 'mywin'})
        vis._send({'data': [trace_loss], 'layout': layout, 'win': 'winloss'})

        outdir_model = pkg_path + '/models'
        if (x % 100 == 0):
            model_name = "turtlebot3_" + str(x)+".json"
            model_name_json_format_path = os.path.join(outdir_model,model_name)
            pickle.dump(qlearn.getQTable(),open(model_name_json_format_path + 'model.pkl', 'wb'))


            #json_object = json.dumps(qlearn.getQTable(), indent = 4)

            #with open(model_name_json_format_path, "w") as json_file:

             #   json_file.write(json_object)
             #   print("Saved model to disk")


            #save(model_name, qlearn.getQTable(), outdir_model)

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

