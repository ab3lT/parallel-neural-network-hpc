/**
 * openmp_train.c - OpenMP parallel implementation of neural network training
 * 
 * Parallelization of Deep Learning Models
 * Distributed Computing for AI - Masters Assignment
 * 
 * Parallelization Strategy: Data Parallelism
 *   - Each thread processes different samples within a mini-batch
 *   - Gradients are accumulated using OpenMP reduction
 *   - Synchronization occurs at batch boundaries for weight updates
 * 
 * Target Architecture: Shared-memory multicore system
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
#include <omp.h>
#include "neural_net.h"

/*============================================================================
 * Thread-local network for parallel processing
 *============================================================================*/

typedef struct {
    NeuralNetwork *net;
    double *grad_weights_accum;  /* Accumulated gradients across layers */
    double *grad_biases_accum;
    int total_grad_weights_size;
    int total_grad_biases_size;
} ThreadLocalNetwork;

ThreadLocalNetwork* create_thread_local_network(NeuralNetwork *base_net) {
    ThreadLocalNetwork *tln = (ThreadLocalNetwork*)malloc(sizeof(ThreadLocalNetwork));
    
    /* Create a copy of the network for thread-local forward/backward pass */
    tln->net = create_network(base_net->learning_rate);
    copy_network(tln->net, base_net);
    
    /* Calculate total gradient sizes */
    tln->total_grad_weights_size = 0;
    tln->total_grad_biases_size = 0;
    for (int l = 0; l < base_net->num_layers; l++) {
        tln->total_grad_weights_size += base_net->layers[l].input_size * 
                                        base_net->layers[l].output_size;
        tln->total_grad_biases_size += base_net->layers[l].output_size;
    }
    
    tln->grad_weights_accum = (double*)calloc(tln->total_grad_weights_size, sizeof(double));
    tln->grad_biases_accum = (double*)calloc(tln->total_grad_biases_size, sizeof(double));
    
    return tln;
}

void free_thread_local_network(ThreadLocalNetwork *tln) {
    free_network(tln->net);
    free(tln->grad_weights_accum);
    free(tln->grad_biases_accum);
    free(tln);
}

void reset_thread_gradients(ThreadLocalNetwork *tln) {
    memset(tln->grad_weights_accum, 0, tln->total_grad_weights_size * sizeof(double));
    memset(tln->grad_biases_accum, 0, tln->total_grad_biases_size * sizeof(double));
    
    for (int l = 0; l < tln->net->num_layers; l++) {
        Layer *layer = &tln->net->layers[l];
        memset(layer->grad_weights, 0, 
               layer->input_size * layer->output_size * sizeof(double));
        memset(layer->grad_biases, 0, layer->output_size * sizeof(double));
    }
}

void accumulate_gradients_to_flat(ThreadLocalNetwork *tln) {
    int w_offset = 0, b_offset = 0;
    for (int l = 0; l < tln->net->num_layers; l++) {
        Layer *layer = &tln->net->layers[l];
        int w_size = layer->input_size * layer->output_size;
        
        for (int i = 0; i < w_size; i++) {
            tln->grad_weights_accum[w_offset + i] += layer->grad_weights[i];
        }
        for (int i = 0; i < layer->output_size; i++) {
            tln->grad_biases_accum[b_offset + i] += layer->grad_biases[i];
        }
        
        w_offset += w_size;
        b_offset += layer->output_size;
    }
}

/*============================================================================
 * Training function - OpenMP Parallel Implementation
 *============================================================================*/

TrainingMetrics* train_openmp(NeuralNetwork *net, Dataset *train_data, 
                               Dataset *test_data, int num_epochs, 
                               int batch_size, int num_threads) {
    TrainingMetrics *metrics = create_metrics(num_epochs);
    
    omp_set_num_threads(num_threads);
    
    int num_batches = (train_data->num_samples + batch_size - 1) / batch_size;
    
    /* Calculate total gradient sizes for reductions */
    int total_grad_w_size = 0, total_grad_b_size = 0;
    for (int l = 0; l < net->num_layers; l++) {
        total_grad_w_size += net->layers[l].input_size * net->layers[l].output_size;
        total_grad_b_size += net->layers[l].output_size;
    }
    
    /* Global gradient accumulators */
    double *global_grad_w = (double*)calloc(total_grad_w_size, sizeof(double));
    double *global_grad_b = (double*)calloc(total_grad_b_size, sizeof(double));
    
    printf("\n=== OpenMP Parallel Training ===\n");
    printf("Number of threads: %d\n", num_threads);
    printf("Training samples: %d\n", train_data->num_samples);
    printf("Test samples: %d\n", test_data->num_samples);
    printf("Batch size: %d\n", batch_size);
    printf("Number of batches: %d\n", num_batches);
    printf("Epochs: %d\n\n", num_epochs);
    
    /* Create thread-local networks */
    ThreadLocalNetwork **thread_nets = (ThreadLocalNetwork**)malloc(
        num_threads * sizeof(ThreadLocalNetwork*));
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        thread_nets[tid] = create_thread_local_network(net);
    }
    
    double total_start = omp_get_wtime();
    
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        double epoch_start = omp_get_wtime();
        double epoch_loss = 0.0;
        int correct = 0;
        
        /* Shuffle training data at the start of each epoch */
        shuffle_dataset(train_data, epoch + 42);
        
        /* Process all batches */
        for (int batch = 0; batch < num_batches; batch++) {
            int batch_start_idx = batch * batch_size;
            int actual_batch_size = batch_size;
            if (batch_start_idx + batch_size > train_data->num_samples) {
                actual_batch_size = train_data->num_samples - batch_start_idx;
            }
            
            /* Reset global gradients */
            memset(global_grad_w, 0, total_grad_w_size * sizeof(double));
            memset(global_grad_b, 0, total_grad_b_size * sizeof(double));
            
            double batch_loss = 0.0;
            int batch_correct = 0;
            
            /* Parallel processing of samples within batch */
            #pragma omp parallel reduction(+:batch_loss,batch_correct)
            {
                int tid = omp_get_thread_num();
                ThreadLocalNetwork *tln = thread_nets[tid];
                
                /* Sync weights from global network */
                copy_network(tln->net, net);
                reset_thread_gradients(tln);
                
                /* Each thread processes a subset of samples */
                #pragma omp for schedule(dynamic)
                for (int s = 0; s < actual_batch_size; s++) {
                    int sample_idx = batch_start_idx + s;
                    double *input = &train_data->data[sample_idx * train_data->input_size];
                    int label = train_data->labels[sample_idx];
                    
                    /* Forward pass */
                    forward_pass(tln->net, input);
                    
                    /* Compute loss */
                    double loss = cross_entropy_loss(
                        tln->net->layers[tln->net->num_layers - 1].output,
                        label, OUTPUT_SIZE
                    );
                    batch_loss += loss;
                    
                    /* Check prediction */
                    int predicted = argmax(
                        tln->net->layers[tln->net->num_layers - 1].output, 
                        OUTPUT_SIZE
                    );
                    if (predicted == label) batch_correct++;
                    
                    /* Backward pass */
                    backward_pass(tln->net, input, label);
                    
                    /* Accumulate to thread-local flat arrays */
                    accumulate_gradients_to_flat(tln);
                    
                    /* Reset layer gradients for next sample */
                    for (int l = 0; l < tln->net->num_layers; l++) {
                        Layer *layer = &tln->net->layers[l];
                        memset(layer->grad_weights, 0, 
                               layer->input_size * layer->output_size * sizeof(double));
                        memset(layer->grad_biases, 0, 
                               layer->output_size * sizeof(double));
                    }
                }
                
                /* Accumulate thread-local gradients to global (with atomic or critical) */
                #pragma omp critical
                {
                    for (int i = 0; i < total_grad_w_size; i++) {
                        global_grad_w[i] += tln->grad_weights_accum[i];
                    }
                    for (int i = 0; i < total_grad_b_size; i++) {
                        global_grad_b[i] += tln->grad_biases_accum[i];
                    }
                }
            }
            
            epoch_loss += batch_loss;
            correct += batch_correct;
            
            /* Average gradients and update weights (sequential) */
            int w_offset = 0, b_offset = 0;
            for (int l = 0; l < net->num_layers; l++) {
                Layer *layer = &net->layers[l];
                int w_size = layer->input_size * layer->output_size;
                
                for (int i = 0; i < w_size; i++) {
                    layer->weights[i] -= net->learning_rate * 
                                        global_grad_w[w_offset + i] / actual_batch_size;
                }
                for (int i = 0; i < layer->output_size; i++) {
                    layer->biases[i] -= net->learning_rate * 
                                       global_grad_b[b_offset + i] / actual_batch_size;
                }
                
                w_offset += w_size;
                b_offset += layer->output_size;
            }
        }
        
        double epoch_time = omp_get_wtime() - epoch_start;
        
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
    
    metrics->total_time = omp_get_wtime() - total_start;
    
    /* Cleanup */
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        free_thread_local_network(thread_nets[tid]);
    }
    free(thread_nets);
    free(global_grad_w);
    free(global_grad_b);
    
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
    int num_threads = omp_get_max_threads();
    
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
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            num_threads = atoi(argv[++i]);
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
            printf("  -t <threads>        Number of OpenMP threads (default: auto)\n");
            printf("  --mnist <dir>       Path to MNIST data directory\n");
            printf("  --train-images <f>  Path to training images file\n");
            printf("  --train-labels <f>  Path to training labels file\n");
            printf("  --test-images <f>   Path to test images file\n");
            printf("  --test-labels <f>   Path to test labels file\n");
            printf("  -h, --help          Show this help message\n");
            return 0;
        }
    }
    
    printf("=================================================\n");
    printf("  OpenMP Parallel Neural Network Training\n");
    printf("=================================================\n");
    printf("\nConfiguration:\n");
    printf("  Network Architecture: %d -> %d -> %d -> %d\n", 
           INPUT_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE, OUTPUT_SIZE);
    printf("  OpenMP threads: %d\n", num_threads);
    
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
            printf("No test data specified, using synthetic test data\n");
            test_data = create_synthetic_dataset(num_samples / 5, INPUT_SIZE, OUTPUT_SIZE, 54321);
        }
    } else {
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
    TrainingMetrics *metrics = train_openmp(net, train_data, test_data, 
                                            num_epochs, batch_size, num_threads);
    
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
    char filename[256];
    snprintf(filename, sizeof(filename), "openmp_%dthreads_metrics.csv", num_threads);
    FILE *fp = fopen(filename, "w");
    if (fp) {
        fprintf(fp, "epoch,loss,accuracy,time\n");
        for (int i = 0; i < metrics->num_epochs; i++) {
            fprintf(fp, "%d,%.6f,%.6f,%.6f\n", i + 1, 
                    metrics->loss_history[i], 
                    metrics->accuracy_history[i],
                    metrics->time_per_epoch[i]);
        }
        fclose(fp);
        printf("\nMetrics saved to %s\n", filename);
    }
    
    /* Cleanup */
    free_metrics(metrics);
    free_network(net);
    free_dataset(train_data);
    free_dataset(test_data);
    
    return 0;
}
