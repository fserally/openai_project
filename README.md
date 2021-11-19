# Installation
Create virtual environment (from Pycharms) and install openAI Gym: pip install gym<br />
Install python 3.9: sudo apt install python3.9<br />

# Installation of ROS Noetic:

1. Setup your computer to accept software from packages.ros.org.<br />
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'<br />
2. Setup your keys<br />
sudo apt install curl <br />
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -<br />
3. Make sure Debian package index is up to date:<br />
sudo apt update<br />
4. Install ROS Noetic (http://wiki.ros.org/noetic/Installation/Ubuntu)<br />
Desktop-Full Install: (Recommended) : Everything in Desktop plus 2D/3D simulators and 2D/3D perception packages<br />
sudo apt install ros-noetic-desktop-full<br />
5. Automatically source this script every time a new shell is launched:<br />
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc<br />
source ~/.bashrc<br />
6. Install dependencies:<br />
sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential<br />
7. Initialize rosdep:<br />
sudo apt install python3-rosdep <br />
sudo rosdep init<br />
rosdep update<br />

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
  

sudo apt install ros-noetic-dynamixel-sdk<br />
sudo apt install ros-noetic-turtlebot3-msgs<br />
sudo apt install ros-noetic-turtlebot3<br />

echo "export TURTLEBOT3_MODEL=burger" >> ~/.bashrc<br />
source ~/.bashrc<br />
Install Turtlebot3 simulation package:<br />
cd ~/catkin_ws/src/<br />
git clone -b noetic-devel https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git<br />
cd ~/catkin_ws && catkin_make<br />

# Installation of OpenAI Ros

Create catkin workspace:<br />
mkdir -p ~/catkin_ws/src<br />
cd ~/catkin_ws/<br />
catkin_make<br />
echo "export TURTLEBOT3_MODEL=burger" >> ~/.bashrc<br />
source ~/.bashrc<br />
echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc<br />
source ~/.bashrc<br />
cd ~/catkin_ws/src<br />
git clone https://github.com/fserally/openai_project.git .<br />

In the virtual environment, install the following dependencies:<br />
pip install defusedxml<br />
pip install pyaml rospkg <br />
pip install torch torchvision <br />
pip install --upgrade cython <br />
pip install tqdm cython pycocotools <br />
pip install matplotlib <br />
pip install opencv-python <br />
pip uninstall em <br />
pip install empy<br />
pip install gitpython<br />
pip install visdom<br />

# Launching the program
In  a terminal, launch the below command to start visdom:<br />
python -m visdom.server<br />
In another terminal, launch the below command to start the training for Turtlebot3:<br />
roslaunch my_turtlebot3_openai_example start_training_v2.launch<br />






