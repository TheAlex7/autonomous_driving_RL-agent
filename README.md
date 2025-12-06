# 271P Project: Autonomous Driving and Pathfinding in a 2D environment.
The goal of this project is to create a Reinforcement Learning Agent with the use of Gymnasium. We decided we wanted to create an agent that can operate a car in a 2D plane and so we thought to modify Box2D's Car Racing in such a way that would reflect a car driving in a grid-like city.

## Custom Gym Environment Quick start
### Step 1 
Install Anaconda Navigator or MiniConda [here.](https://www.anaconda.com/download)

### Step 2
Install our custom conda environment.
```
conda env create -f environment.yml
```
### Step 3
Activate the conda environment.
```
conda activate autodriving2d
```
### Step 4
Run ```run_env.py``` to confirm the conda and gymnasium environments are running properly. You may also look through this file to see an example on how to access our gymnasium environment.