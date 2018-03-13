import sys
import os
import csv
import math
import numpy as np
import random


def run(train_input, val_input, train_out, val_out, metrics, num_epoch,
        hidden_units, flag, step):
    # convert input to matrix/vector
    train = parsing(train_input)
    train_data = train[0]
    train_labels = train[1]
    val = parsing(val_input)
    val_data = val[0]
    val_labels = val[1]

    # init a and b
    labels_num = 10
    feats_num = 129
    matrix_a = init_weights(hidden_units, feats_num, flag)
    matrix_b = init_weights(labels_num, hidden_units + 1, flag)

    output_str = ""

    # update using SGD
    for epoch in range(num_epoch):
        for i in range(len(train_labels)):
            # compute neural network layer
            for_res = forward(train_data[i], train_labels[i],
                              matrix_a, matrix_b)
            # compute gradients via backprop
            back_res = back(train_data[i], matrix_b, for_res)
            # update matrix a and b
            matrix_a = matrix_a - step * back_res[0]
            matrix_b = matrix_b - step * back_res[1]
        # eval training entropy
        train_entropy = get_entropy(train_data, train_labels,
                                    matrix_a, matrix_b)
        str_1 = "epoch={} crossentropy(train): {}\n".format(epoch + 1,
                                                            train_entropy)
        # eval validation entropy
        val_entropy = get_entropy(val_data, val_labels, matrix_a, matrix_b)
        str_2 = "epoch={} crossentropy(validation): {}\n".format(epoch + 1,
                                                                 val_entropy)
        output_str += str_1
        output_str += str_2

    # test
    train_res = test(train_data, train_labels, matrix_a, matrix_b)
    str_train = "error(train): {}\n".format(train_res[1])
    output_str += str_train
    val_res = test(val_data, val_labels, matrix_a, matrix_b)
    str_val = "error(validation): {}\n".format(val_res[1])
    output_str += str_val

    # write into files
    metrics_out = open(metrics, 'w')
    metrics_out.write(output_str)

    train_file = open(train_out, 'w')
    train_file.write(train_res[0])

    val_file = open(val_out, 'w')
    val_file.write(val_res[0])

    metrics_out.close()
    train_file.close()
    val_file.close()

    return None


####################################
######## Helper function ###########


def parsing(input_file):
    feats = list()
    labels = list()
    with open(input_file, 'r') as tsv:
        for line in csv.reader(tsv):
            labels.append(int(line.pop(0)))
            line = list(map(int, line))
            line.insert(0, 1)   # insert bias term at beginning
            feats.append(np.array(line))
            #feats.append(line)
    #print(len(feats[0]))
    #print(len(labels))
    result = [feats, labels]
    return result


def init_weights(width, height, flag):
    if (flag == 1):
        matrix = np.random.uniform(-0.1, 0.1, (width, height))
        bias = np.zeros(width)
        matrix[:, 0] = bias
    elif (flag == 2):
        matrix = np.zeros((width, height))
    return matrix    # numpy matrix


def forward(feat, label, matrix_a, matrix_b):
    #print(feat)
    #print(np.dot(feat, feat))
    feat = np.transpose(np.array([feat]))
    #print(feat)
    a = np.matmul(matrix_a, feat)  # numpy array
    sigmoid_func = np.vectorize(sigmoid)
    z = sigmoid_func(a)
    z = np.insert(z, 0, 1)
    b = np.matmul(matrix_b, z)
    denominator = np.sum(np.exp(b))
    y_hat = np.divide(np.exp(b), denominator)
    y_label = np.array([0] * 10)
    np.put(y_label, [label], 1)
    j = -1 * np.dot(y_label, np.log(y_hat))
    #return [a, z, b, y_hat, y_label, j]
    return [z, y_hat, y_label, j]


def back(feat, matrix_b, pre_res):
    z = pre_res[0]
    y_hat = pre_res[1]
    y_label = pre_res[2]
    print("y_hat: ", y_hat)
    gy = -1 * np.divide(y_label, y_hat)
    print("gy: ", gy)
    square_matrix = sq_matrix(y_hat)
    print("sq_matrix: ", square_matrix)
    gb = np.matmul(gy, square_matrix)
    print("gb: ", gb)
    gb = np.transpose(np.array([gb]))
    print("gb: ", gb)
    gBeta = np.matmul(gb, np.array([z]))
    print("gBeta: ", gBeta)
    gz = np.matmul(np.transpose(matrix_b), gb)
    print("gz: ", gz)
    #print("gz: ", np.transpose(gz))
    #print("multi: ", np.multiply(z, 1-z))
    ga = np.multiply(np.transpose(gz), np.multiply(z, 1-z))
    ga = np.transpose(ga)
    ga = np.delete(ga, 0, 0)
    print("ga: ", ga)
    gAlpha = np.matmul(ga, np.array([feat]))
    #print("gAlpha: ", gAlpha)

    return [gAlpha, gBeta]


def get_entropy(feats, labels, matrix_a, matrix_b):
    res = 0
    for i in range(len(feats)):
        j = forward(feats[i], labels[i], matrix_a, matrix_b)
        res += j[3]
    return res / len(feats)


def test(feats, labels, matrix_a, matrix_b):
    count = 0
    predicted = ""
    for i in range(len(feats)):
        feat = feats[i]
        label = labels[i]
        a = np.matmul(matrix_a, feat)  # numpy array
        sigmoid_func = np.vectorize(sigmoid)
        z = sigmoid_func(a)
        z = np.insert(z, 0, 1)
        b = np.matmul(matrix_b, z)
        denominator = np.sum(np.exp(b))
        y_hat = np.divide(np.exp(b), denominator)
        y_predicted = np.argmax(y_hat)
        predicted += str(y_predicted)
        predicted += "\n"
        if (y_predicted != label):
            count += 1
    rate = float(count) / len(labels)
    return [predicted, rate]


def sigmoid(x):
    y = 1 / (1 + math.exp(-1 * x))
    return y


def sq_matrix(vec):
    temp1 = np.diag(vec)
    temp2 = np.matmul(np.transpose(np.array([vec])), np.array([vec]))
    return temp1 - temp2

####################################
######## Main function #############


if __name__ == '__main__':
    train_input = sys.argv[1]           #csv file
    val_input = sys.argv[2]             #csv file
    train_out = sys.argv[3]             #.labels
    val_out = sys.argv[4]               #.labels
    metrics = sys.argv[5]               #txt file
    num_epoch = int(sys.argv[6])        #integer
    hidden_units = int(sys.argv[7])     #integer
    init_flag = int(sys.argv[8])        #1 or 2
    learning_rate = float(sys.argv[9])  #float value

    run(train_input, val_input, train_out, val_out, metrics, num_epoch,
        hidden_units, init_flag, learning_rate)