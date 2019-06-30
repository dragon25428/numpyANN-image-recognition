import numpy as np
import skimage.io, skimage.color, skimage.feature
import os
import seaborn as sns
import matplotlib.pyplot as plt

# extract features
fruits = ["apple", "raspberry", "mango", "lemon"]
#492+490+490+490=1,962
dataset_features = np.zeros(shape=(1962, 360))
outputs = np.zeros(shape=(1962))

idx = 0
class_label = 0
for fruit_dir in fruits:
    path = "C:\\Users\\Wilson\\Downloads\\NumPyANN-master\\NumPyANN-master"
    all_imgs = os.listdir(path+"\\"+fruit_dir)
    for img_file in all_imgs:
        if img_file.endswith(".jpg"): # Ensures reading only JPG files.
            fruit_data = skimage.io.imread(fname=path+"\\"+fruit_dir+'\\'+img_file, as_gray=False)
            print("loading {}".format(fruit_dir))
            fruit_data_hsv = skimage.color.rgb2hsv(rgb=fruit_data)
            hist, bin_edges = np.histogram(a=fruit_data_hsv[:, :, 0], bins=360)
            dataset_features[idx, :] = hist
            outputs[idx] = class_label
            idx = idx + 1
    class_label = class_label + 1

# Model
def sigmoid(inpt):
    return 1.0 / (1 + np.exp(-1 * inpt))

def relu(inpt):
    result = inpt
    result[inpt < 0] = 0
    return result

def update_weights(weights, learning_rate):
    new_weights = weights - learning_rate * weights
    return new_weights

def train_network(num_iterations, weights, data_inputs, data_outputs, learning_rate, activation="relu"):
    for iteration in range(num_iterations):
        print("Itreation ", iteration)
        for sample_idx in range(data_inputs.shape[0]):
            r1 = data_inputs[sample_idx, :]
            for idx in range(len(weights) - 1):
                curr_weights = weights[idx]
                r1 = np.matmul(r1, curr_weights)
                if activation == "relu":
                    r1 = relu(r1)
                elif activation == "sigmoid":
                    r1 = sigmoid(r1)
            curr_weights = weights[-1]
            r1 = np.matmul(r1, curr_weights)
            predicted_label = np.where(r1 == np.max(r1))[0][0]
            desired_label = data_outputs[sample_idx]
            if predicted_label != desired_label:
                weights = update_weights(weights,
                                         learning_rate=0.001)
    return weights

def predict_outputs(weights, data_inputs, activation="relu"):
    predictions = np.zeros(shape=(data_inputs.shape[0]))
    for sample_idx in range(data_inputs.shape[0]):
        r1 = data_inputs[sample_idx, :]
        for curr_weights in weights:
            r1 = np.matmul(r1, curr_weights)
            if activation == "relu":
                r1 = relu(r1)
            elif activation == "sigmoid":
                r1 = sigmoid(r1)
        predicted_label = np.where(r1 == np.max(r1))[0][0]
        predictions[sample_idx] = predicted_label
    return predictions

features_STDs = np.std(a=dataset_features, axis=0)
data_inputs = dataset_features[:, features_STDs > 50]
data_outputs = outputs

HL1_neurons = 150
input_HL1_weights = np.random.uniform(low=-0.1, high=0.1,
                                         size=(data_inputs.shape[1], HL1_neurons))
HL2_neurons = 60
HL1_HL2_weights = np.random.uniform(low=-0.1, high=0.1,
                                       size=(HL1_neurons, HL2_neurons))
output_neurons = 4
HL2_output_weights = np.random.uniform(low=-0.1, high=0.1,
                                          size=(HL2_neurons, output_neurons))

weights = np.array([input_HL1_weights,
                       HL1_HL2_weights,
                       HL2_output_weights])

weights = train_network(num_iterations=100,
                        weights=weights,
                        data_inputs=data_inputs,
                        data_outputs=data_outputs,
                        learning_rate=0.001)

predictions = predict_outputs(weights, data_inputs)
Accuracy = sum(predictions==data_outputs) / len(predictions)