# Semi-Autonomous Continuous Robotic Arm Control Using an Augmented Reality Brain-Computer Interface

## Overview
This project contains the software for continously controlling a [Reachy](https://pollen-robotics.github.io/reachy-2021-docs/) robotic arm (Pollen Robotics, France) using an augmented reality (AR) brain-computer interface (BCI). The system displays five flashing stimuli using a HoloLens 2 (Microsoft, USA) in a cross-pattern around the robot end-effector. The HoloLens and robot coordinate systems are aligned using a [QR code](https://github.com/microsoft/MixedReality-QRCode-Sample) in the scene. Electroencephalograpy (EEG) data gathered via a [lab streaming layer](https://labstreaminglayer.org/#/) are continuously decoded using canonical correlation analysis to determine which stimulus the user is attending to, and translated into a directional BCI control command. 

The session manager can be used to run an observation trial or a robotic reaching trial. In observation trials, the arm moves in each of the five directions and brain activity is recorded but only used to control the robot if feedback is enabled. In reaching trials, the system can use direct control (DC), where BCI commands control end-effector translation, or shared control (SC). In SC, user inputs are blended with an assistance vector that pulls the end-effector towards the system's prediction of which object the user is trying to reach.

## Files
- **session_manager.py** master GUI for running an experiment and communicating with the EEG, HoloLens and Reachy.
- **obj_pos.py** testing script for controling the robot to reach to set locations in space.
- **session_results.py** analysis notebook for extracting data from a recording (stored using [LabRecorder](https://github.com/labstreaminglayer/App-LabRecorder).
- **exp_results.py** notebook for analysis of experiment results across multiple sessions.
- **offline_analysis.py** notebook for analysing/decoding EEG data in a recording.
- **decoding.py** classes/functions for processing online EEG data and offline recordings.
- **robot_control.py** classes/functions to control the Reachy robot and shared control algorithms.
- **stimulus.py** classes/functions to control stimuli parameters by communicating with the HoloLens.
- **Training_session** folder: Stimulus presentation server developed using Unity 2022.3. **StimBehaviour.cs** and **StimServer.cs** control the stimuli and communicate with the Python client.
