# IEEE_TKDE
Ensemble works using PSO and Class based clustering 

If you would like to use these please refer the following paper:

Z.Jan, J.C. Munoz, and A.Ali, "A Novel Method for Creating an Optimized Ensemble Classifier by Introducing Cluster Size Reduction and Diversity", IEEE Transactions on Knowledge and Data Engineering, 2020. 

1. In order to run the code use the file mainProgram.m. 
2. The datasets are passed into the variable Problem as a structure of strings.
3. The value of K is run for 2 to 10, which can be changed to any number as required. But do note that very high values of K will end up creating data clusters that will contain only 1 or 2 samples and therefore, will cause training issues with classifiers.
4. The resuls over 10-fold are saved in a csv file automatically. 
5. The list of classifiers is given as a structure of classifier names. 
6. If anyone would like to add more classifiers this can be done by editing the function trainClassifiers.
7. Particle Swarm Optimization is used from the global optimization toolbox of MATLAB therefore, make sure you have the toolbox before running the code.
8. TrainNetwork is also used to train Neural networks which are a part of Image Processing Toolbox in Matlab therefore, that should also be installed.

Thank you.
