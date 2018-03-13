You could get handwritten letters labeled using Neural Network by running the following command

$ python neuralnet_matmul.py [args...]

Where [args...] is a placeholder for 9 command-line aguments:
1. <train_input> : path to training input .csv file
2. <validation_input> : path to validation input .scv file
3. <train_out> : path to output .labels file to which the prediction on training data will be written
4. <validation_out> : path to output .labels file to which the prediction on validation data will be written
5. <metrics_out> : path of output .txt file to which train error and validation error will be wwritten
6. <num_epoch> : positive integer specifying the number of times backpropogation loops through all of the training data
7. <hidden_units> : positive integer specifying the number of hidden units
8. <init_flag> : integer taking value 1 or 2 specifying to use RANDOM or ZERO initialization
9. <learning_rate> : float value specifying the learning rate for SGD

For example:
$ python neuralnet_matmul.py smalltrain.csv smallvalidation.csv train_out.labels val_out.labels metrics.txt 2 4 2 0.1


