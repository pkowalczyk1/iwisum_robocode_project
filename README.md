# Robocode DQN project
## Setup instruction:
1. Add robocode dependencies to local maven repository using `robocode_maven_import.sh` script (assuming that you have maven installed)
2. In `.idea/runConfigurations/Robocode.xml` change the `option name="WORKING_DIRECTORY"` to point to wherever your robocode is installed (relatively to the project directory)
3. In directory iwisum_robocode_project run using python 3.11 or lower
```
    pip install tensorflow
    pip install tf_agents
    python robot_driver.py 
```
4. Try to run the application using the IntelliJ run configuration (`Robocode`) - everything should work, and you should see the `QLearningRobot` in robocode