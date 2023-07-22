# InstrumentationProject
### In this repository there are some of the scripts and files used that I used in my final instrumentation project during my Electrical Engineering course at UFRGS.

## Objective
The objective of this project was to design and simulate an electronic measurement system. 
The system was designed to measure temperature, pressure and molar fraction of natural gas samples.
With those three quantities measured, a Multi-Layer Perceptron (MLP) Neural Network was trained to predict the mass density of the gas sample. 

The data was gathered from:
Wood, D. A. Choubineh, A. Transparent machine learning provides insightful estimates of natural
gas density based on pressure, temperature and compositional variables. 2020. Avaiable in:
https://www.sciencedirect.com/science/article/pii/S2468256X20300031.

For complete information about the project, it's development, analysis and evaluation, the file "Project Report.pdf" can be read.
If there are still any doubts, please contact me.
My e-mail is: bcaetani.poa@gmail.com

## Files explanation
1. molecular_weight_data.xlsx file is the dataset used in NN training.
2. neuralNetwork.py was used to optimize the network hyperparameters and perform, training, validation and testing.
3. supportFunction.py was used to analyse and design pt100 temperature measurement circuit desired voltage output and to perform linear regression of the voltage as a function of the measured temperature, using data contained in "vout vs temp.xlsx" file.
 
