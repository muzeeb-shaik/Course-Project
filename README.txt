######################## CSCE 633 Homework 4 Submission #############################

This code uses Python Version 3.7.4 so run the code using Python 3.

Pandas, numpy, seaborn, Matplotlib, Statistics, Graphviz, Time, Psutil, tabulate, researchpy, sciki-learn and keras need to be preinstalled

The files "cnn2.py", "FFNN Covid 2 Class.py" can be directly run to get the 2 class classification result for FFNN and CNN.
The files "cnn3.py", "FFNN Covid 3 Class.py" can be directly run to get the 3 class classification result for FFNN and CNN.

NOTE - The above code assumes that folders - "Covid64", "Normal64" and "Pneumonia64" are already present in the current working directory. The folder contains scaled 64X64 images.

========================================================================================================================================
CSCE633_S20_Ananya_Gargi_Keerat_Muzeeb_Mythili_Project

** The folder structure **
>>>>COVID-19 Radiography Database : Comprises the complete dataset used to run the image
>>>>Covid64, Normal64 and Pneumonia64 : Comprises scaled 64X64 images for covid, normal and pnemonia patients in respective folders.  
>>>>CNN_output, FFNN_output : Consists of output result images(graphs) for CNN and FFNN code respectively. 
>>>>Covid64_changed_orient, Normal64_changed_orient, Pneumonia64_changed_orient : Comprises scaled, rotated and mirrored (created from original dataset) 64X64 images for covid, normal and pnemonia patients in respective folders.  

** The code files **
>>>>cnn2.py, FFNN Covid 2 Class.py,cnn3.py, FFNN Covid 3 Class.py : Main python code files.
>>>>feature_extraction.py : The code responsible for generating the 64X64 images and also the textfile comprising image pixel values in vector form.
>>>>McNemar Test.py : File for Mcnemar test. Input for the files - "predicted_values_cnn.txt" and "predicted_values_ffnn.txt" for 2 class classification. "predicted_values_cnn_3class.txt" and "predicted_values_ffnn_3class.txt" for 3 class classification.

** For Web Application using Flask **
>>>> Install all dependencies - flask, numpy, Werkzeug, keras, gevent, h5py, tensorflow
>>>> From 'WebApplication' directory run 'python app.py' file 
>>>> Open http://localhost:5000 for viewing the tool in a browser
>>>> Access this Github repository from right corner of web application
>>>> To use a different model, update keras model in 'models' folder and update 'app.py' file to load that particular model.
>>>> The current tool uses ImageDataGenerator in Keras to pre-process the uploaded image. Modify 'model_predict' function to process data compatible to your saved model. 
>>>> Run 'Ctrl + C' to stop the server running


========================================================================================================================================

** Default Hyperparameters **

Input Image =  64X64        
Batch Size = 128 
Epoch = 20
Loss function = Categorical Cross Entropy
Optimizer  = Adam		
Activation function =  Relu

The parameters can be changed by going into the respective location and changing the parameters. 

The perceptron and decision tree code used in the result is not shown.
The main model runs with parameters - 

© 2020 GitHub, Inc.
