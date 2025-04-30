# Overview

This repository is a collection of scripts used used to analyze brain-computer interface (BCI) data (behavior, spiking data) from decoder perturbation tasks: visuomotor rotation (VMR), shuffle.

This code was used to analyze data and generate plots for the following manuscripts:


	paper 1: "Neural population variance explains adaptation differences during learning" (Stealey et al., under review 2024) https://doi.org/10.1101/2024.10.01.616115
		rotation perturbation studies only


	paper 2: "Population-level constraints on single neuron tuning changes during behavioral adaptation" (Stealey et al., under review 2025) https://doi.org/10.1101/2025.04.17.649401
		rotation and shuffle perturbation studies



For task code, see https://github.com/santacruzlab  (branch: bmi_ssd) bmi_python>built_in_tasks>error_clamp_tasks.py: class BMICursorVisRotErrorClamp. (for rotation perturbation task)

# System requirements 
## Hardware Requirements
standard computer (no non-standard harware)
## Software Requirements 
made and tested on Windows 11 (version 24H2; OS build 26100.3476) using Spyder (5.1.5) and Python 3.9.7.

## Python Dependencies
	os
	pickle
	numpy
	pandas
	scipy
	scikit-learn
	statsmodels
	seaborn
	matplotlib
	tqdm
	datetime

For complete list with versions, view "requirements.txt".



# Installation Guide
 "Installation" is quick a process. You can either clone the repository using your preferred method (e.g., HTTPS, SSH, GitHubCLI) or download (and unzip) a ZIP folder of the repository.  To ensure that all of the dependencies are installed, install "requirements.txt".  To do this, in a command terminal using  the 'cd' command, navigate to the directory in which you saved the "bci_analysis" repository. Then enter the following command:

	pip install -r requirements.txt





# Demo
This respository is designed for the user to run individual scripts and largely relies on implementation of existing Python libraries. See "Data Processing Scripts" below for more information on how to run each script on the sample sessions (DEMO_data; < 1.5GB of raw data from sessions of different perturbation types). NOTE: Prior to running each script, change the "root_path" to reflect the full file path of where "bci_analysis" was saved.  In general, each script produces a .pkl file that contains (a) Pandas dataframe(s) (and sometimes additional variables). 



# Reproduction Instructions
FUTURE (once the (preprocessed) data are fully published): The plots in paper 2 can be reproduced by running the corresponding scripts under bci_analysis\2_plot_and_compute_stats\paper2. Prior to running each script, update the code in each file to reflect the complete file path (variable: "root_path") to your local "bci_analysis" location.

# License
BSD 3-Clause License (see LICENSE.txt for more details)



# Data Processing Scripts


#### GENERAL PREPROCESSING
**bci_analysis\1_processing\main_process_hdf.py**
	-DATA PROCESSING STARTS WITH THIS FILE. 
	-Main script for extracting neural and behavioral data and related decoder information for each BCI session (more details in file). 
	-Running this file will iterate through all sessions. Extracted results are saved in Pandas dataframes and saved to .pkl files. 
	-RUNTIME on DEMO_data: < 2 minutes total


**bci_analysis\1_processing\main_get_trial_inds.py**
	-Saves the trials indices (and their lengths) for baseline (block 1) and perturbation (block 2) blocks to Pandas dataframes (.pkl files). 
	-Determines the minimum length of a trial within the baselien and perturbation blocks.
	-Ensures that the minimum number of trials needed exist.
	-RUNTIME on DEMO_data: < 10 seconds total
	  


#### BEHAVIOR
**bci_analysis\1_processing\2_behavior\main_behavior.py**
	-Calculates behavioral metrics presented in the paper (trial time and distance) as well as additional behavioral metrics (angular error, movement error, movement variability).
	-RUNTIME on DEMO_data: < 15 seconds total


#### NEURAL ANALYSIS - VARIANCE/FACTOR ANALYSIS (FA) for paper 1
**bci_analysis\1_preprocessing\3_FA\main_FA_TTT.py**
	-Performs factor analysis on spiking data (1 baseline block model, 1 perturbation block model). The input is the number of trials by the number of neurons. Each entry is the number of spikes over a fixed 900ms window for neuron, i, on trial, t.
	-Script contains a series of steps: (1) data formatting, (2) cross-validation, (3) fit models, (4) compute metrics.  
	-Not used for paper 2 (Population-level constraints on single neuron..)


#### NEURAL ANALYSIS - VARIANCE/FACTOR ANALYSIS (FA) for paper 2
**bci_analysis\1_preprocessing\3_FA\main_FA_TTT_for_tuning.py**
	-Very similar to main_FA_TTT.py, except only the baseline model is used in later analyses and the window to count spikes per neuron & trial is different (to match tuning methods).
	-RUNTIME on DEMO_data: (1) data formating, < 5 seconds total; (2) cross-validation < 2 minutes total ; (3) fit models & (4) compute metrics < 5 seconds total



#### NEURAL ANALYSIS - TUNING CURVE for paper 2
**bci_analysis\1_preprocessing\3_FA\main_tuning_curve.py**
	-Fit cosine tuning curves to each BCI neuron and assess significance using bootstrapping techniques.
	-RUNTIME on DEMO_data: (1) generate boostrap means, about 3 minutes total; (2) fit cosine tuning curves, > 80 minutes total (45 minutes for Brazos, rotation (many neurons)); (3) calculate changes in preferred direction, < 20 seconds total




# Common Abbreviations
BCI: brain-comptuer interface  
BMI: brain-machine interface (used interchangeably with BCI)  
BL: baseline (block)  
PE: perturbation (block)  
PL: second (or "late") perturbation (block)  
WO: washout (block)  



#### Updates
04.29.2025: additional documentation