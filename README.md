In this repository there are some tools and python scripts that I used for the modelling and prediction of the ionization energy and atomic energy of some set of atoms back in 2021. In the folder old, there are the scripts that I originally used to initially predict this properties with a multiple linear model and random forest regressor. In the script model_regressor.py the function estimator_model creates a model of any sklearn estimator such as LinearRegression or/and RandomForestRegressor() and others. An idea of the objective of this project is shown below:


![alt text](pictures/DNN_DIAGRAM_.jpg?raw=true) 

The script AM.py contains a class and a set of python instructions to construcut a pandas dataframe from the three different files produced from total, $\alpha$ and $\beta$ calculations. 
The database that was constructed for this project is in the file IE_AE_DNN_2.csv, the python script keras_dnn_IEAE.py makes a cross validation model of a neural network with 2 layers and the results of these models are in the outpouts folders.
