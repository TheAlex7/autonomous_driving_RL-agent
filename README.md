# Autonomous Driving agent taught with Reinforcement Learning

Project done by [@TheAlex7](https://github.com/TheAlex7), [@KaelynTaing](https://github.com/KaelynTaing), [@richZ-Z](https://github.com/richZ-Z/), and [@ninjadare](https://github.com/ninjadare) for AI 271P of the UCI MCS program.
Custom Environment based off Box2D's CarRacing simulator compatible with OpenAI's Gymnasium API
Demo/Intro Video: https://www.youtube.com/watch?v=hiqOwT89EEY

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
```
python run_env.py
```