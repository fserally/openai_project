# Installation
Create virtual environment (from Pycharms) and install openAI Gym: pip install gym
Install python 3.9: sudo apt install python3.9

# Installation of ROS Noetic:

1. Setup your computer to accept software from packages.ros.org.
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
2. Setup your keys
sudo apt install curl 
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
3. Make sure Debian package index is up to date:
sudo apt update
4. Install ROS Noetic (http://wiki.ros.org/noetic/Installation/Ubuntu)
Desktop-Full Install: (Recommended) : Everything in Desktop plus 2D/3D simulators and 2D/3D perception packages
sudo apt install ros-noetic-desktop-full
5. Automatically source this script every time a new shell is launched:
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source ~/.bashrc
6. Install dependencies:
sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
7. Initialize rosdep:
sudo apt install python3-rosdep 
sudo rosdep init
rosdep update

# Installation of Turtlebot3 packages

# Install turtlebot3 packages for ROS Noetic

sudo apt-get install ros-noetic-joy ros-noetic-teleop-twist-joy \
  ros-noetic-teleop-twist-keyboard ros-noetic-laser-proc \
  ros-noetic-rgbd-launch ros-noetic-rosserial-arduino \
  ros-noetic-rosserial-python ros-noetic-rosserial-client \
  ros-noetic-rosserial-msgs ros-noetic-amcl ros-noetic-map-server \
  ros-noetic-move-base ros-noetic-urdf ros-noetic-xacro \
  ros-noetic-compressed-image-transport ros-noetic-rqt* ros-noetic-rviz \
  ros-noetic-gmapping ros-noetic-navigation ros-noetic-interactive-markers
  

sudo apt install ros-noetic-dynamixel-sdk
sudo apt install ros-noetic-turtlebot3-msgs
sudo apt install ros-noetic-turtlebot3

echo "export TURTLEBOT3_MODEL=burger" >> ~/.bashrc
source ~/.bashrc
Install Turtlebot3 simulation package:
cd ~/catkin_ws/src/
git clone -b noetic-devel https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git
cd ~/catkin_ws && catkin_make

# Installation of OpenAI Ros

Create catkin workspace:
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/
catkin_make
echo "export TURTLEBOT3_MODEL=burger" >> ~/.bashrc
source ~/.bashrc
echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc
cd ~/catkin_ws/src
git clone https://github.com/fserally/openai_project.git .

In the virtual environment, install the following dependencies:
pip install defusedxml
pip install pyaml rospkg 
pip install torch torchvision 
pip install --upgrade cython 
pip install tqdm cython pycocotools 
pip install matplotlib 
pip install opencv-python 
pip uninstall em 
pip install empy
pip install gitpython
pip install visdom

# Launching the program
In  a terminal, launch the below command to start visdom:
python -m visdom.server
In another terminal, launch the below command to start the training for Turtlebot3:
roslaunch my_turtlebot3_openai_example start_training_v2.launch






