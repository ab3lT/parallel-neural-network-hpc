/**
 * serial_train.c - Serial baseline implementation of neural network training
 * 
 * Parallelization of Deep Learning Models
 * Distributed Computing for AI - Masters Assignment
 * 
 * This implementation serves as the baseline for performance comparison
 * with parallel implementations.
 * 
 * Architecture:
 *   - Input Layer:  784 neurons (28x28 image)
 *   - Hidden 1:     256 neurons (ReLU activation)
 *   - Hidden 2:     128 neurons (ReLU activation)
 *   - Output:       10 neurons (Softmax activation)
 * 
 * Loss Function: Cross-Entropy
 * Optimizer: Mini-batch Stochastic Gradient Descent (SGD)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "neural_net.h"

/*============================================================================
 * Training function - Serial Implementation
 *============================================================================*/

TrainingMetrics* train_serial(NeuralNetwork *net, Dataset *train_data, 
                               Dataset *test_data, int num_epochs, 
                               int batch_size) {
    TrainingMetrics *metrics = create_metrics(num_epochs);
    
    int num_batches = (train_data->num_samples + batch_size - 1) / batch_size;
    
    printf("\n=== Serial Training ===\n");
    printf("Training samples: %d\n", train_data->num_samples);
    printf("Test samples: %d\n", test_data->num_samples);
    printf("Batch size: %d\n", batch_size);
    printf("Number of batches: %d\n", num_batches);
    printf("Epochs: %d\n\n", num_epochs);
    
    double total_start = get_time();
    
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        double epoch_start = get_time();
        double epoch_loss = 0.0;
        int correct = 0;
        
        /* Shuffle training data at the start of each epoch */
        shuffle_dataset(train_data, epoch + 42);
        
        /* Process all batches */
        for (int batch = 0; batch < num_batches; batch++) {
            int batch_start = batch * batch_size;
            int actual_batch_size = batch_size;
            if (batch_start + batch_size > train_data->num_samples) {
                actual_batch_size = train_data->num_samples - batch_start;
            }
            
            /* Reset gradients for this batch */
            for (int l = 0; l < net->num_layers; l++) {
                Layer *layer = &net->layers[l];
                memset(layer->grad_weights, 0, 
                       layer->input_size * layer->output_size * sizeof(double));
                memset(layer->grad_biases, 0, 
                       layer->output_size * sizeof(double));
            }
            
            /* Process each sample in the batch */
            for (int s = 0; s < actual_batch_size; s++) {
                int sample_idx = batch_start + s;
                double *input = &train_data->data[sample_idx * train_data->input_size];
                int label = train_data->labels[sample_idx];
                
                /* Forward pass */
                forward_pass(net, input);
                
                /* Compute loss */
                double loss = cross_entropy_loss(
                    net->layers[net->num_layers - 1].output, 
                    label, 
                    OUTPUT_SIZE
                );
                epoch_loss += loss;
                
                /* Check prediction */
                int predicted = argmax(net->layers[net->num_layers - 1].output, OUTPUT_SIZE);
                if (predicted == label) correct++;
                
                /* Backward pass (accumulate gradients) */
                backward_pass(net, input, label);
            }
            
            /* Average gradients over batch */
            for (int l = 0; l < net->num_layers; l++) {
                Layer *layer = &net->layers[l];
                for (int i = 0; i < layer->input_size * layer->output_size; i++) {
                    layer->grad_weights[i] /= actual_batch_size;
                }
                for (int j = 0; j < layer->output_size; j++) {
                    layer->grad_biases[j] /= actual_batch_size;
                }
            }
            
            /* Update weights */
            update_weights(net);
        }
        
        double epoch_time = get_time() - epoch_start;
        
        /* Record metrics */
        metrics->loss_history[epoch] = epoch_loss / train_data->num_samples;
        metrics->accuracy_history[epoch] = (double)correct / train_data->num_samples;
        metrics->time_per_epoch[epoch] = epoch_time;
        metrics->num_epochs = epoch + 1;
        
        /* Compute test accuracy */
        double test_accuracy = compute_accuracy(net, test_data);
        
        printf("Epoch %3d/%d | Loss: %.6f | Train Acc: %.2f%% | Test Acc: %.2f%% | Time: %.3fs\n",
               epoch + 1, num_epochs,
               metrics->loss_history[epoch],
               metrics->accuracy_history[epoch] * 100,
               test_accuracy * 100,
               epoch_time);
    }
    
    metrics->total_time = get_time() - total_start;
    
    return metrics;
}

/*============================================================================
 * Main function
 *============================================================================*/

int main(int argc, char *argv[]) {
    /* Default parameters */
    int num_samples = 10000;
    int num_epochs = 20;
    int batch_size = DEFAULT_BATCH_SIZE;
    double learning_rate = DEFAULT_LEARNING_RATE;
    
    /* MNIST paths (NULL = use synthetic data) */
    char *mnist_dir = NULL;
    char *train_images = NULL;
    char *train_labels = NULL;
    char *test_images = NULL;
    char *test_labels = NULL;
    
    /* Parse command line arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            num_samples = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-e") == 0 && i + 1 < argc) {
            num_epochs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) {
            batch_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-lr") == 0 && i + 1 < argc) {
            learning_rate = atof(argv[++i]);
        } else if (strcmp(argv[i], "--mnist") == 0 && i + 1 < argc) {
            mnist_dir = argv[++i];
        } else if (strcmp(argv[i], "--train-images") == 0 && i + 1 < argc) {
            train_images = argv[++i];
        } else if (strcmp(argv[i], "--train-labels") == 0 && i + 1 < argc) {
            train_labels = argv[++i];
        } else if (strcmp(argv[i], "--test-images") == 0 && i + 1 < argc) {
            test_images = argv[++i];
        } else if (strcmp(argv[i], "--test-labels") == 0 && i + 1 < argc) {
            test_labels = argv[++i];
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  -n <samples>        Number of training samples (default: 10000)\n");
            printf("  -e <epochs>         Number of epochs (default: 20)\n");
            printf("  -b <batch>          Batch size (default: 64)\n");
            printf("  -lr <rate>          Learning rate (default: 0.01)\n");
            printf("  --mnist <dir>       Path to MNIST data directory\n");
            printf("  --train-images <f>  Path to training images file\n");
            printf("  --train-labels <f>  Path to training labels file\n");
            printf("  --test-images <f>   Path to test images file\n");
            printf("  --test-labels <f>   Path to test labels file\n");
            printf("  -h, --help          Show this help message\n");
            printf("\nMNIST Usage:\n");
            printf("  %s --mnist ./data/mnist -e 20\n", argv[0]);
            printf("  (expects train-images-idx3-ubyte, train-labels-idx1-ubyte, etc.)\n");
            return 0;
        }
    }
    
    printf("=================================================\n");
    printf("  Serial Neural Network Training (Baseline)\n");
    printf("=================================================\n");
    printf("\nConfiguration:\n");
    printf("  Network Architecture: %d -> %d -> %d -> %d\n", 
           INPUT_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE, OUTPUT_SIZE);
    
    Dataset *train_data = NULL;
    Dataset *test_data = NULL;
    
    /* Build MNIST paths if directory is specified */
    char train_img_path[512], train_lbl_path[512];
    char test_img_path[512], test_lbl_path[512];
    
    if (mnist_dir != NULL) {
        snprintf(train_img_path, sizeof(train_img_path), "%s/train-images-idx3-ubyte", mnist_dir);
        snprintf(train_lbl_path, sizeof(train_lbl_path), "%s/train-labels-idx1-ubyte", mnist_dir);
        snprintf(test_img_path, sizeof(test_img_path), "%s/t10k-images-idx3-ubyte", mnist_dir);
        snprintf(test_lbl_path, sizeof(test_lbl_path), "%s/t10k-labels-idx1-ubyte", mnist_dir);
        train_images = train_img_path;
        train_labels = train_lbl_path;
        test_images = test_img_path;
        test_labels = test_lbl_path;
    }
    
    /* Load MNIST or generate synthetic data */
    if (train_images != NULL && train_labels != NULL) {
        printf("\nLoading MNIST training data...\n");
        train_data = load_mnist(train_images, train_labels, num_samples);
        if (train_data == NULL) {
            fprintf(stderr, "Failed to load MNIST training data\n");
            return 1;
        }
        
        if (test_images != NULL && test_labels != NULL) {
            printf("Loading MNIST test data...\n");
            test_data = load_mnist(test_images, test_labels, num_samples / 5);
            if (test_data == NULL) {
                fprintf(stderr, "Failed to load MNIST test data\n");
                free_dataset(train_data);
                return 1;
            }
        } else {
            /* Use part of training data for testing */
            printf("No test data specified, using synthetic test data\n");
            test_data = create_synthetic_dataset(num_samples / 5, INPUT_SIZE, OUTPUT_SIZE, 54321);
        }
    } else {
        /* Use synthetic data */
        printf("\nGenerating synthetic dataset...\n");
        train_data = create_synthetic_dataset(num_samples, INPUT_SIZE, OUTPUT_SIZE, 12345);
        test_data = create_synthetic_dataset(num_samples / 5, INPUT_SIZE, OUTPUT_SIZE, 54321);
    }
    
    printf("  Training samples: %d\n", train_data->num_samples);
    printf("  Test samples: %d\n", test_data->num_samples);
    printf("  Epochs: %d\n", num_epochs);
    printf("  Batch size: %d\n", batch_size);
    printf("  Learning rate: %.4f\n", learning_rate);
    
    /* Create and initialize network */
    printf("Initializing neural network...\n");
    NeuralNetwork *net = create_network(learning_rate);
    initialize_weights(net, 42);
    
    /* Train the network */
    TrainingMetrics *metrics = train_serial(net, train_data, test_data, 
                                            num_epochs, batch_size);
    
    /* Print final summary */
    printf("\n=================================================\n");
    printf("  Training Complete\n");
    printf("=================================================\n");
    printf("Total training time: %.3f seconds\n", metrics->total_time);
    printf("Average time per epoch: %.3f seconds\n", 
           metrics->total_time / metrics->num_epochs);
    printf("Final training loss: %.6f\n", 
           metrics->loss_history[metrics->num_epochs - 1]);
    printf("Final training accuracy: %.2f%%\n", 
           metrics->accuracy_history[metrics->num_epochs - 1] * 100);
    printf("Final test accuracy: %.2f%%\n", 
           compute_accuracy(net, test_data) * 100);
    
    /* Save metrics to file for comparison */
    FILE *fp = fopen("serial_metrics.csv", "w");
    if (fp) {
        fprintf(fp, "epoch,loss,accuracy,time\n");
        for (int i = 0; i < metrics->num_epochs; i++) {
            fprintf(fp, "%d,%.6f,%.6f,%.6f\n", i + 1, 
                    metrics->loss_history[i], 
                    metrics->accuracy_history[i],
                    metrics->time_per_epoch[i]);
        }
        fclose(fp);
        printf("\nMetrics saved to serial_metrics.csv\n");
    }
    
    /* Cleanup */
    free_metrics(metrics);
    free_network(net);
    free_dataset(train_data);
    free_dataset(test_data);
    
    return 0;
}
