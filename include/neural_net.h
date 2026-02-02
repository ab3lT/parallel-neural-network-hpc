/**
 * neural_net.h - Common header for neural network structures and utilities
 * 
 * Parallelization of Deep Learning Models
 * Distributed Computing for AI - Masters Assignment
 */

#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* Network Configuration */
#define INPUT_SIZE 784      /* 28x28 images (MNIST-like) */
#define HIDDEN1_SIZE 256    /* First hidden layer */
#define HIDDEN2_SIZE 128    /* Second hidden layer */
#define OUTPUT_SIZE 10      /* 10 classes (digits 0-9) */

#define MAX_EPOCHS 50
#define DEFAULT_BATCH_SIZE 64
#define DEFAULT_LEARNING_RATE 0.01

/* Activation function types */
typedef enum {
    ACTIVATION_RELU,
    ACTIVATION_SIGMOID,
    ACTIVATION_SOFTMAX
} ActivationType;

/* Layer structure */
typedef struct {
    int input_size;
    int output_size;
    double *weights;        /* input_size x output_size */
    double *biases;         /* output_size */
    double *output;         /* output_size (activations) */
    double *z;              /* output_size (pre-activation) */
    double *grad_weights;   /* Gradients for weights */
    double *grad_biases;    /* Gradients for biases */
    double *delta;          /* Error signal for backprop */
    ActivationType activation;
} Layer;

/* Neural Network structure */
typedef struct {
    int num_layers;
    Layer *layers;
    double learning_rate;
} NeuralNetwork;

/* Dataset structure */
typedef struct {
    double *data;           /* num_samples x input_size */
    int *labels;            /* num_samples */
    int num_samples;
    int input_size;
    int num_classes;
} Dataset;

/* Training metrics */
typedef struct {
    double *loss_history;
    double *accuracy_history;
    double *time_per_epoch;
    int num_epochs;
    double total_time;
} TrainingMetrics;

/* Function prototypes - Network operations */
NeuralNetwork* create_network(double learning_rate);
void free_network(NeuralNetwork *net);
void initialize_weights(NeuralNetwork *net, unsigned int seed);
void copy_network(NeuralNetwork *dest, NeuralNetwork *src);

/* Forward and backward pass */
void forward_pass(NeuralNetwork *net, double *input);
void backward_pass(NeuralNetwork *net, double *input, int label);
void update_weights(NeuralNetwork *net);

/* Activation functions */
double relu(double x);
double relu_derivative(double x);
double sigmoid(double x);
double sigmoid_derivative(double x);
void softmax(double *input, double *output, int size);

/* Loss functions */
double cross_entropy_loss(double *output, int label, int num_classes);

/* Dataset operations */
Dataset* create_synthetic_dataset(int num_samples, int input_size, int num_classes, unsigned int seed);
Dataset* load_mnist(const char *images_path, const char *labels_path, int max_samples);
void free_dataset(Dataset *data);
void shuffle_dataset(Dataset *data, unsigned int seed);
void get_batch(Dataset *data, int batch_start, int batch_size, 
               double *batch_data, int *batch_labels);

/* Utility functions */
double get_time(void);
int argmax(double *arr, int size);
double compute_accuracy(NeuralNetwork *net, Dataset *data);

/* Metrics */
TrainingMetrics* create_metrics(int max_epochs);
void free_metrics(TrainingMetrics *metrics);

#endif /* NEURAL_NET_H */
