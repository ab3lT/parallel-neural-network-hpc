/**
 * utils.c - Common utility functions for neural network
 * 
 * Parallelization of Deep Learning Models
 * Distributed Computing for AI - Masters Assignment
 */

#define _USE_MATH_DEFINES
#include "neural_net.h"
#include <sys/time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*============================================================================
 * Time measurement
 *============================================================================*/

double get_time(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

/*============================================================================
 * Activation functions
 *============================================================================*/

double relu(double x) {
    return x > 0 ? x : 0;
}

double relu_derivative(double x) {
    return x > 0 ? 1.0 : 0.0;
}

double sigmoid(double x) {
    if (x > 500) return 1.0;
    if (x < -500) return 0.0;
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    double s = sigmoid(x);
    return s * (1.0 - s);
}

void softmax(double *input, double *output, int size) {
    /* Find max for numerical stability */
    double max_val = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max_val) max_val = input[i];
    }
    
    /* Compute exp and sum */
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        output[i] = exp(input[i] - max_val);
        sum += output[i];
    }
    
    /* Normalize */
    for (int i = 0; i < size; i++) {
        output[i] /= sum;
    }
}

/*============================================================================
 * Loss functions
 *============================================================================*/

double cross_entropy_loss(double *output, int label, int num_classes) {
    double prob = output[label];
    if (prob < 1e-15) prob = 1e-15;  /* Prevent log(0) */
    return -log(prob);
}

/*============================================================================
 * Network creation and management
 *============================================================================*/

Layer* create_layer(int input_size, int output_size, ActivationType activation) {
    Layer *layer = (Layer*)malloc(sizeof(Layer));
    layer->input_size = input_size;
    layer->output_size = output_size;
    layer->activation = activation;
    
    /* Allocate memory */
    layer->weights = (double*)calloc(input_size * output_size, sizeof(double));
    layer->biases = (double*)calloc(output_size, sizeof(double));
    layer->output = (double*)calloc(output_size, sizeof(double));
    layer->z = (double*)calloc(output_size, sizeof(double));
    layer->grad_weights = (double*)calloc(input_size * output_size, sizeof(double));
    layer->grad_biases = (double*)calloc(output_size, sizeof(double));
    layer->delta = (double*)calloc(output_size, sizeof(double));
    
    return layer;
}

void free_layer(Layer *layer) {
    free(layer->weights);
    free(layer->biases);
    free(layer->output);
    free(layer->z);
    free(layer->grad_weights);
    free(layer->grad_biases);
    free(layer->delta);
    free(layer);
}

NeuralNetwork* create_network(double learning_rate) {
    NeuralNetwork *net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    net->learning_rate = learning_rate;
    net->num_layers = 3;
    
    /* Allocate layer array */
    net->layers = (Layer*)malloc(net->num_layers * sizeof(Layer));
    
    /* Layer 1: Input -> Hidden1 (ReLU) */
    Layer *l1 = create_layer(INPUT_SIZE, HIDDEN1_SIZE, ACTIVATION_RELU);
    net->layers[0] = *l1;
    free(l1);
    
    /* Layer 2: Hidden1 -> Hidden2 (ReLU) */
    Layer *l2 = create_layer(HIDDEN1_SIZE, HIDDEN2_SIZE, ACTIVATION_RELU);
    net->layers[1] = *l2;
    free(l2);
    
    /* Layer 3: Hidden2 -> Output (Softmax) */
    Layer *l3 = create_layer(HIDDEN2_SIZE, OUTPUT_SIZE, ACTIVATION_SOFTMAX);
    net->layers[2] = *l3;
    free(l3);
    
    return net;
}

void free_network(NeuralNetwork *net) {
    for (int i = 0; i < net->num_layers; i++) {
        free(net->layers[i].weights);
        free(net->layers[i].biases);
        free(net->layers[i].output);
        free(net->layers[i].z);
        free(net->layers[i].grad_weights);
        free(net->layers[i].grad_biases);
        free(net->layers[i].delta);
    }
    free(net->layers);
    free(net);
}

void initialize_weights(NeuralNetwork *net, unsigned int seed) {
    srand(seed);
    
    for (int l = 0; l < net->num_layers; l++) {
        Layer *layer = &net->layers[l];
        
        /* He initialization for ReLU, Xavier for others */
        double scale;
        if (layer->activation == ACTIVATION_RELU) {
            scale = sqrt(2.0 / layer->input_size);  /* He initialization */
        } else {
            scale = sqrt(1.0 / layer->input_size);  /* Xavier initialization */
        }
        
        for (int i = 0; i < layer->input_size * layer->output_size; i++) {
            /* Box-Muller transform for normal distribution */
            double u1 = (rand() + 1.0) / (RAND_MAX + 2.0);
            double u2 = (rand() + 1.0) / (RAND_MAX + 2.0);
            double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
            layer->weights[i] = z * scale;
        }
        
        /* Initialize biases to zero */
        for (int i = 0; i < layer->output_size; i++) {
            layer->biases[i] = 0.0;
        }
    }
}

void copy_network(NeuralNetwork *dest, NeuralNetwork *src) {
    dest->learning_rate = src->learning_rate;
    dest->num_layers = src->num_layers;
    
    for (int l = 0; l < src->num_layers; l++) {
        Layer *sl = &src->layers[l];
        Layer *dl = &dest->layers[l];
        
        memcpy(dl->weights, sl->weights, sl->input_size * sl->output_size * sizeof(double));
        memcpy(dl->biases, sl->biases, sl->output_size * sizeof(double));
    }
}

/*============================================================================
 * Forward pass
 *============================================================================*/

void forward_pass(NeuralNetwork *net, double *input) {
    double *current_input = input;
    
    for (int l = 0; l < net->num_layers; l++) {
        Layer *layer = &net->layers[l];
        
        /* Compute z = W * input + b */
        for (int j = 0; j < layer->output_size; j++) {
            layer->z[j] = layer->biases[j];
            for (int i = 0; i < layer->input_size; i++) {
                layer->z[j] += current_input[i] * layer->weights[i * layer->output_size + j];
            }
        }
        
        /* Apply activation function */
        if (layer->activation == ACTIVATION_RELU) {
            for (int j = 0; j < layer->output_size; j++) {
                layer->output[j] = relu(layer->z[j]);
            }
        } else if (layer->activation == ACTIVATION_SIGMOID) {
            for (int j = 0; j < layer->output_size; j++) {
                layer->output[j] = sigmoid(layer->z[j]);
            }
        } else if (layer->activation == ACTIVATION_SOFTMAX) {
            softmax(layer->z, layer->output, layer->output_size);
        }
        
        current_input = layer->output;
    }
}

/*============================================================================
 * Backward pass
 *============================================================================*/

void backward_pass(NeuralNetwork *net, double *input, int label) {
    /* Output layer delta (softmax + cross-entropy) */
    Layer *output_layer = &net->layers[net->num_layers - 1];
    for (int j = 0; j < output_layer->output_size; j++) {
        output_layer->delta[j] = output_layer->output[j];
    }
    output_layer->delta[label] -= 1.0;  /* Gradient of softmax + CE */
    
    /* Backpropagate through layers */
    for (int l = net->num_layers - 1; l >= 0; l--) {
        Layer *layer = &net->layers[l];
        double *prev_output = (l > 0) ? net->layers[l-1].output : input;
        
        /* Compute gradients for weights and biases */
        for (int j = 0; j < layer->output_size; j++) {
            layer->grad_biases[j] += layer->delta[j];
            for (int i = 0; i < layer->input_size; i++) {
                layer->grad_weights[i * layer->output_size + j] += 
                    prev_output[i] * layer->delta[j];
            }
        }
        
        /* Compute delta for previous layer (if not input layer) */
        if (l > 0) {
            Layer *prev_layer = &net->layers[l-1];
            for (int i = 0; i < layer->input_size; i++) {
                double sum = 0.0;
                for (int j = 0; j < layer->output_size; j++) {
                    sum += layer->weights[i * layer->output_size + j] * layer->delta[j];
                }
                /* Apply activation derivative */
                if (prev_layer->activation == ACTIVATION_RELU) {
                    prev_layer->delta[i] = sum * relu_derivative(prev_layer->z[i]);
                } else if (prev_layer->activation == ACTIVATION_SIGMOID) {
                    prev_layer->delta[i] = sum * sigmoid_derivative(prev_layer->z[i]);
                } else {
                    prev_layer->delta[i] = sum;
                }
            }
        }
    }
}

/*============================================================================
 * Weight updates
 *============================================================================*/

void update_weights(NeuralNetwork *net) {
    for (int l = 0; l < net->num_layers; l++) {
        Layer *layer = &net->layers[l];
        
        for (int i = 0; i < layer->input_size * layer->output_size; i++) {
            layer->weights[i] -= net->learning_rate * layer->grad_weights[i];
            layer->grad_weights[i] = 0.0;  /* Reset gradient */
        }
        
        for (int j = 0; j < layer->output_size; j++) {
            layer->biases[j] -= net->learning_rate * layer->grad_biases[j];
            layer->grad_biases[j] = 0.0;  /* Reset gradient */
        }
    }
}

/*============================================================================
 * Dataset operations
 *============================================================================*/

Dataset* create_synthetic_dataset(int num_samples, int input_size, int num_classes, unsigned int seed) {
    Dataset *data = (Dataset*)malloc(sizeof(Dataset));
    data->num_samples = num_samples;
    data->input_size = input_size;
    data->num_classes = num_classes;
    
    data->data = (double*)malloc(num_samples * input_size * sizeof(double));
    data->labels = (int*)malloc(num_samples * sizeof(int));
    
    srand(seed);
    
    /* Generate synthetic data with some structure */
    for (int s = 0; s < num_samples; s++) {
        int label = s % num_classes;
        data->labels[s] = label;
        
        /* Create class-specific patterns */
        for (int i = 0; i < input_size; i++) {
            double noise = (rand() / (double)RAND_MAX - 0.5) * 0.3;
            /* Different patterns for different classes */
            double signal = 0.0;
            
            /* Create distinct patterns for each class */
            int region = i / (input_size / num_classes);
            if (region == label) {
                signal = 0.8;  /* High activation in class-specific region */
            } else {
                signal = 0.2;  /* Low activation elsewhere */
            }
            
            data->data[s * input_size + i] = signal + noise;
            
            /* Clamp to [0, 1] */
            if (data->data[s * input_size + i] < 0) data->data[s * input_size + i] = 0;
            if (data->data[s * input_size + i] > 1) data->data[s * input_size + i] = 1;
        }
    }
    
    return data;
}

/*----------------------------------------------------------------------------
 * MNIST Dataset Loading
 * 
 * MNIST file format (IDX):
 * - Images: [magic(4)][count(4)][rows(4)][cols(4)][pixels...]
 * - Labels: [magic(4)][count(4)][labels...]
 * All values are big-endian
 *----------------------------------------------------------------------------*/

static int read_int_be(FILE *fp) {
    unsigned char bytes[4];
    if (fread(bytes, 1, 4, fp) != 4) return -1;
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

Dataset* load_mnist(const char *images_path, const char *labels_path, int max_samples) {
    FILE *img_fp = fopen(images_path, "rb");
    FILE *lbl_fp = fopen(labels_path, "rb");
    
    if (!img_fp) {
        fprintf(stderr, "Error: Cannot open images file: %s\n", images_path);
        if (lbl_fp) fclose(lbl_fp);
        return NULL;
    }
    if (!lbl_fp) {
        fprintf(stderr, "Error: Cannot open labels file: %s\n", labels_path);
        fclose(img_fp);
        return NULL;
    }
    
    /* Read image file header */
    int img_magic = read_int_be(img_fp);
    int num_images = read_int_be(img_fp);
    int rows = read_int_be(img_fp);
    int cols = read_int_be(img_fp);
    
    /* Read label file header */
    int lbl_magic = read_int_be(lbl_fp);
    int num_labels = read_int_be(lbl_fp);
    
    /* Validate magic numbers */
    if (img_magic != 2051) {
        fprintf(stderr, "Error: Invalid image file magic number: %d (expected 2051)\n", img_magic);
        fclose(img_fp);
        fclose(lbl_fp);
        return NULL;
    }
    if (lbl_magic != 2049) {
        fprintf(stderr, "Error: Invalid label file magic number: %d (expected 2049)\n", lbl_magic);
        fclose(img_fp);
        fclose(lbl_fp);
        return NULL;
    }
    
    /* Validate counts match */
    if (num_images != num_labels) {
        fprintf(stderr, "Error: Image count (%d) != label count (%d)\n", num_images, num_labels);
        fclose(img_fp);
        fclose(lbl_fp);
        return NULL;
    }
    
    printf("MNIST: Found %d images of size %dx%d\n", num_images, rows, cols);
    
    /* Limit samples if requested */
    int num_samples = num_images;
    if (max_samples > 0 && max_samples < num_images) {
        num_samples = max_samples;
        printf("MNIST: Limiting to %d samples\n", num_samples);
    }
    
    int input_size = rows * cols;  /* Should be 784 for MNIST */
    
    /* Allocate dataset */
    Dataset *data = (Dataset*)malloc(sizeof(Dataset));
    data->num_samples = num_samples;
    data->input_size = input_size;
    data->num_classes = 10;
    
    data->data = (double*)malloc(num_samples * input_size * sizeof(double));
    data->labels = (int*)malloc(num_samples * sizeof(int));
    
    if (!data->data || !data->labels) {
        fprintf(stderr, "Error: Memory allocation failed for MNIST dataset\n");
        free(data->data);
        free(data->labels);
        free(data);
        fclose(img_fp);
        fclose(lbl_fp);
        return NULL;
    }
    
    /* Read pixel data */
    unsigned char *pixel_buffer = (unsigned char*)malloc(input_size);
    for (int i = 0; i < num_samples; i++) {
        if (fread(pixel_buffer, 1, input_size, img_fp) != (size_t)input_size) {
            fprintf(stderr, "Error: Failed to read image %d\n", i);
            free(pixel_buffer);
            free_dataset(data);
            fclose(img_fp);
            fclose(lbl_fp);
            return NULL;
        }
        
        /* Normalize pixels to [0, 1] */
        for (int j = 0; j < input_size; j++) {
            data->data[i * input_size + j] = pixel_buffer[j] / 255.0;
        }
    }
    free(pixel_buffer);
    
    /* Read labels */
    unsigned char *label_buffer = (unsigned char*)malloc(num_samples);
    if (fread(label_buffer, 1, num_samples, lbl_fp) != (size_t)num_samples) {
        fprintf(stderr, "Error: Failed to read labels\n");
        free(label_buffer);
        free_dataset(data);
        fclose(img_fp);
        fclose(lbl_fp);
        return NULL;
    }
    
    for (int i = 0; i < num_samples; i++) {
        data->labels[i] = (int)label_buffer[i];
    }
    free(label_buffer);
    
    fclose(img_fp);
    fclose(lbl_fp);
    
    printf("MNIST: Successfully loaded %d samples\n", num_samples);
    
    return data;
}

void free_dataset(Dataset *data) {
    free(data->data);
    free(data->labels);
    free(data);
}

void shuffle_dataset(Dataset *data, unsigned int seed) {
    srand(seed);
    
    /* Fisher-Yates shuffle */
    for (int i = data->num_samples - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        
        /* Swap samples */
        for (int k = 0; k < data->input_size; k++) {
            double temp = data->data[i * data->input_size + k];
            data->data[i * data->input_size + k] = data->data[j * data->input_size + k];
            data->data[j * data->input_size + k] = temp;
        }
        
        /* Swap labels */
        int temp_label = data->labels[i];
        data->labels[i] = data->labels[j];
        data->labels[j] = temp_label;
    }
}

void get_batch(Dataset *data, int batch_start, int batch_size, 
               double *batch_data, int *batch_labels) {
    int actual_size = batch_size;
    if (batch_start + batch_size > data->num_samples) {
        actual_size = data->num_samples - batch_start;
    }
    
    memcpy(batch_data, &data->data[batch_start * data->input_size], 
           actual_size * data->input_size * sizeof(double));
    memcpy(batch_labels, &data->labels[batch_start], 
           actual_size * sizeof(int));
}

/*============================================================================
 * Utility functions
 *============================================================================*/

int argmax(double *arr, int size) {
    int max_idx = 0;
    double max_val = arr[0];
    for (int i = 1; i < size; i++) {
        if (arr[i] > max_val) {
            max_val = arr[i];
            max_idx = i;
        }
    }
    return max_idx;
}

double compute_accuracy(NeuralNetwork *net, Dataset *data) {
    int correct = 0;
    
    for (int i = 0; i < data->num_samples; i++) {
        forward_pass(net, &data->data[i * data->input_size]);
        int predicted = argmax(net->layers[net->num_layers - 1].output, OUTPUT_SIZE);
        if (predicted == data->labels[i]) {
            correct++;
        }
    }
    
    return (double)correct / data->num_samples;
}

/*============================================================================
 * Metrics
 *============================================================================*/

TrainingMetrics* create_metrics(int max_epochs) {
    TrainingMetrics *metrics = (TrainingMetrics*)malloc(sizeof(TrainingMetrics));
    metrics->loss_history = (double*)calloc(max_epochs, sizeof(double));
    metrics->accuracy_history = (double*)calloc(max_epochs, sizeof(double));
    metrics->time_per_epoch = (double*)calloc(max_epochs, sizeof(double));
    metrics->num_epochs = 0;
    metrics->total_time = 0.0;
    return metrics;
}

void free_metrics(TrainingMetrics *metrics) {
    free(metrics->loss_history);
    free(metrics->accuracy_history);
    free(metrics->time_per_epoch);
    free(metrics);
}

void print_metrics(TrainingMetrics *metrics) {
    printf("\n=== Training Summary ===\n");
    printf("Total epochs: %d\n", metrics->num_epochs);
    printf("Total time: %.3f seconds\n", metrics->total_time);
    printf("Average time per epoch: %.3f seconds\n", 
           metrics->total_time / metrics->num_epochs);
    printf("\nFinal loss: %.6f\n", metrics->loss_history[metrics->num_epochs - 1]);
    printf("Final accuracy: %.2f%%\n", 
           metrics->accuracy_history[metrics->num_epochs - 1] * 100);
}
