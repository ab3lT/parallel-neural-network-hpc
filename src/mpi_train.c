/**
 * mpi_train.c - MPI parallel implementation of neural network training
 * 
 * Parallelization of Deep Learning Models
 * Distributed Computing for AI - Masters Assignment
 * 
 * Parallelization Strategy: Data Parallelism with Distributed Memory
 *   - Training data is partitioned across MPI processes
 *   - Each process computes gradients on its local data subset
 *   - Gradients are synchronized using MPI_Allreduce
 *   - All processes maintain consistent model weights
 * 
 * Target Architecture: Distributed-memory cluster system
 * 
 * Communication Pattern:
 *   - MPI_Allreduce for gradient averaging (collective operation)
 *   - Synchronous updates at batch boundaries
 * 
 * Architecture:
 *   - Input Layer:  784 neurons (28x28 image)
 *   - Hidden 1:     256 neurons (ReLU activation)
 *   - Hidden 2:     128 neurons (ReLU activation)
 *   - Output:       10 neurons (Softmax activation)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include "neural_net.h"

/*============================================================================
 * MPI-specific data distribution
 *============================================================================*/

typedef struct {
    double *local_data;
    int *local_labels;
    int local_num_samples;
    int global_num_samples;
} LocalDataset;

LocalDataset* distribute_dataset(Dataset *global_data, int rank, int size) {
    LocalDataset *local = (LocalDataset*)malloc(sizeof(LocalDataset));
    local->global_num_samples = global_data->num_samples;
    
    /* Calculate local data distribution */
    int base_samples = global_data->num_samples / size;
    int remainder = global_data->num_samples % size;
    
    int start_idx, end_idx;
    if (rank < remainder) {
        local->local_num_samples = base_samples + 1;
        start_idx = rank * (base_samples + 1);
    } else {
        local->local_num_samples = base_samples;
        start_idx = remainder * (base_samples + 1) + (rank - remainder) * base_samples;
    }
    end_idx = start_idx + local->local_num_samples;
    
    /* Allocate and copy local data */
    local->local_data = (double*)malloc(local->local_num_samples * 
                                        global_data->input_size * sizeof(double));
    local->local_labels = (int*)malloc(local->local_num_samples * sizeof(int));
    
    memcpy(local->local_data, 
           &global_data->data[start_idx * global_data->input_size],
           local->local_num_samples * global_data->input_size * sizeof(double));
    memcpy(local->local_labels,
           &global_data->labels[start_idx],
           local->local_num_samples * sizeof(int));
    
    return local;
}

void free_local_dataset(LocalDataset *local) {
    free(local->local_data);
    free(local->local_labels);
    free(local);
}

void shuffle_local_data(LocalDataset *local, int input_size, unsigned int seed, int rank) {
    srand(seed + rank);
    
    for (int i = local->local_num_samples - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        
        /* Swap samples */
        for (int k = 0; k < input_size; k++) {
            double temp = local->local_data[i * input_size + k];
            local->local_data[i * input_size + k] = local->local_data[j * input_size + k];
            local->local_data[j * input_size + k] = temp;
        }
        
        /* Swap labels */
        int temp_label = local->local_labels[i];
        local->local_labels[i] = local->local_labels[j];
        local->local_labels[j] = temp_label;
    }
}

/*============================================================================
 * Training function - MPI Parallel Implementation
 *============================================================================*/

TrainingMetrics* train_mpi(NeuralNetwork *net, LocalDataset *local_train,
                           Dataset *test_data, int num_epochs,
                           int batch_size, int rank, int size) {
    TrainingMetrics *metrics = create_metrics(num_epochs);
    
    /* Calculate total gradient sizes */
    int total_grad_w_size = 0, total_grad_b_size = 0;
    for (int l = 0; l < net->num_layers; l++) {
        total_grad_w_size += net->layers[l].input_size * net->layers[l].output_size;
        total_grad_b_size += net->layers[l].output_size;
    }
    int total_grad_size = total_grad_w_size + total_grad_b_size;
    
    /* Allocate gradient buffers */
    double *local_grads = (double*)calloc(total_grad_size, sizeof(double));
    double *global_grads = (double*)calloc(total_grad_size, sizeof(double));
    
    int local_num_batches = (local_train->local_num_samples + batch_size - 1) / batch_size;
    
    if (rank == 0) {
        printf("\n=== MPI Parallel Training ===\n");
        printf("Number of processes: %d\n", size);
        printf("Total training samples: %d\n", local_train->global_num_samples);
        printf("Local samples (rank 0): %d\n", local_train->local_num_samples);
        printf("Test samples: %d\n", test_data->num_samples);
        printf("Batch size: %d\n", batch_size);
        printf("Local batches per epoch: %d\n", local_num_batches);
        printf("Epochs: %d\n\n", num_epochs);
    }
    
    double total_start = MPI_Wtime();
    
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        double epoch_start = MPI_Wtime();
        double local_epoch_loss = 0.0;
        int local_correct = 0;
        
        /* Shuffle local data */
        shuffle_local_data(local_train, INPUT_SIZE, epoch + 42, rank);
        
        /* Process all local batches */
        for (int batch = 0; batch < local_num_batches; batch++) {
            int batch_start_idx = batch * batch_size;
            int actual_batch_size = batch_size;
            if (batch_start_idx + batch_size > local_train->local_num_samples) {
                actual_batch_size = local_train->local_num_samples - batch_start_idx;
            }
            
            /* Reset local gradients */
            memset(local_grads, 0, total_grad_size * sizeof(double));
            for (int l = 0; l < net->num_layers; l++) {
                Layer *layer = &net->layers[l];
                memset(layer->grad_weights, 0, 
                       layer->input_size * layer->output_size * sizeof(double));
                memset(layer->grad_biases, 0, layer->output_size * sizeof(double));
            }
            
            /* Process samples in batch */
            for (int s = 0; s < actual_batch_size; s++) {
                int sample_idx = batch_start_idx + s;
                double *input = &local_train->local_data[sample_idx * INPUT_SIZE];
                int label = local_train->local_labels[sample_idx];
                
                /* Forward pass */
                forward_pass(net, input);
                
                /* Compute loss */
                double loss = cross_entropy_loss(
                    net->layers[net->num_layers - 1].output,
                    label, OUTPUT_SIZE
                );
                local_epoch_loss += loss;
                
                /* Check prediction */
                int predicted = argmax(net->layers[net->num_layers - 1].output, OUTPUT_SIZE);
                if (predicted == label) local_correct++;
                
                /* Backward pass */
                backward_pass(net, input, label);
            }
            
            /* Flatten gradients into local_grads buffer */
            int offset = 0;
            for (int l = 0; l < net->num_layers; l++) {
                Layer *layer = &net->layers[l];
                int w_size = layer->input_size * layer->output_size;
                
                memcpy(&local_grads[offset], layer->grad_weights, w_size * sizeof(double));
                offset += w_size;
            }
            for (int l = 0; l < net->num_layers; l++) {
                Layer *layer = &net->layers[l];
                memcpy(&local_grads[offset], layer->grad_biases, 
                       layer->output_size * sizeof(double));
                offset += layer->output_size;
            }
            
            /* Allreduce to sum gradients across all processes */
            MPI_Allreduce(local_grads, global_grads, total_grad_size,
                         MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            
            /* Get total batch size across all processes */
            int total_batch_size;
            MPI_Allreduce(&actual_batch_size, &total_batch_size, 1,
                         MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            
            /* Update weights with averaged gradients */
            offset = 0;
            for (int l = 0; l < net->num_layers; l++) {
                Layer *layer = &net->layers[l];
                int w_size = layer->input_size * layer->output_size;
                
                for (int i = 0; i < w_size; i++) {
                    layer->weights[i] -= net->learning_rate * 
                                        global_grads[offset + i] / total_batch_size;
                }
                offset += w_size;
            }
            for (int l = 0; l < net->num_layers; l++) {
                Layer *layer = &net->layers[l];
                for (int i = 0; i < layer->output_size; i++) {
                    layer->biases[i] -= net->learning_rate * 
                                       global_grads[offset + i] / total_batch_size;
                }
                offset += layer->output_size;
            }
        }
        
        /* Reduce loss and correct count across all processes */
        double global_epoch_loss;
        int global_correct;
        MPI_Reduce(&local_epoch_loss, &global_epoch_loss, 1, MPI_DOUBLE, 
                   MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_correct, &global_correct, 1, MPI_INT, 
                   MPI_SUM, 0, MPI_COMM_WORLD);
        
        double epoch_time = MPI_Wtime() - epoch_start;
        
        /* Record metrics (only on rank 0) */
        if (rank == 0) {
            metrics->loss_history[epoch] = global_epoch_loss / local_train->global_num_samples;
            metrics->accuracy_history[epoch] = (double)global_correct / local_train->global_num_samples;
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
    }
    
    if (rank == 0) {
        metrics->total_time = MPI_Wtime() - total_start;
    }
    
    /* Cleanup */
    free(local_grads);
    free(global_grads);
    
    return metrics;
}

/*============================================================================
 * Main function
 *============================================================================*/

int main(int argc, char *argv[]) {
    int rank, size;
    
    /* Initialize MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
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
        } else if ((strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) && rank == 0) {
            printf("Usage: mpirun -np <procs> %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  -n <samples>        Number of training samples (default: 10000)\n");
            printf("  -e <epochs>         Number of epochs (default: 20)\n");
            printf("  -b <batch>          Batch size per process (default: 64)\n");
            printf("  -lr <rate>          Learning rate (default: 0.01)\n");
            printf("  --mnist <dir>       Path to MNIST data directory\n");
            printf("  --train-images <f>  Path to training images file\n");
            printf("  --train-labels <f>  Path to training labels file\n");
            printf("  --test-images <f>   Path to test images file\n");
            printf("  --test-labels <f>   Path to test labels file\n");
            printf("  -h, --help          Show this help message\n");
            MPI_Finalize();
            return 0;
        }
    }
    
    if (rank == 0) {
        printf("=================================================\n");
        printf("  MPI Parallel Neural Network Training\n");
        printf("=================================================\n");
        printf("\nConfiguration:\n");
        printf("  Network Architecture: %d -> %d -> %d -> %d\n", 
               INPUT_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE, OUTPUT_SIZE);
        printf("  MPI processes: %d\n", size);
    }
    
    Dataset *global_train = NULL;
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
    /* All processes load the same data (in real app, rank 0 would load and distribute) */
    if (train_images != NULL && train_labels != NULL) {
        if (rank == 0) printf("\nLoading MNIST training data...\n");
        global_train = load_mnist(train_images, train_labels, num_samples);
        if (global_train == NULL) {
            if (rank == 0) fprintf(stderr, "Failed to load MNIST training data\n");
            MPI_Finalize();
            return 1;
        }
        
        if (test_images != NULL && test_labels != NULL) {
            if (rank == 0) printf("Loading MNIST test data...\n");
            test_data = load_mnist(test_images, test_labels, num_samples / 5);
            if (test_data == NULL) {
                if (rank == 0) fprintf(stderr, "Failed to load MNIST test data\n");
                free_dataset(global_train);
                MPI_Finalize();
                return 1;
            }
        } else {
            if (rank == 0) printf("No test data specified, using synthetic test data\n");
            test_data = create_synthetic_dataset(num_samples / 5, INPUT_SIZE, OUTPUT_SIZE, 54321);
        }
    } else {
        if (rank == 0) printf("\nGenerating synthetic dataset...\n");
        global_train = create_synthetic_dataset(num_samples, INPUT_SIZE, OUTPUT_SIZE, 12345);
        test_data = create_synthetic_dataset(num_samples / 5, INPUT_SIZE, OUTPUT_SIZE, 54321);
    }
    
    if (rank == 0) {
        printf("  Training samples: %d\n", global_train->num_samples);
        printf("  Test samples: %d\n", test_data->num_samples);
        printf("  Epochs: %d\n", num_epochs);
        printf("  Batch size: %d\n", batch_size);
        printf("  Learning rate: %.4f\n", learning_rate);
    }
    
    /* Distribute training data */
    LocalDataset *local_train = distribute_dataset(global_train, rank, size);
    
    if (rank == 0) {
        printf("Data distributed across %d processes\n", size);
    }
    
    /* Create and initialize network (same on all processes) */
    if (rank == 0) printf("Initializing neural network...\n");
    NeuralNetwork *net = create_network(learning_rate);
    initialize_weights(net, 42);
    
    /* Synchronize initial weights */
    MPI_Barrier(MPI_COMM_WORLD);
    
    /* Train the network */
    TrainingMetrics *metrics = train_mpi(net, local_train, test_data, 
                                         num_epochs, batch_size, rank, size);
    
    /* Print final summary (rank 0 only) */
    if (rank == 0) {
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
        
        /* Save metrics to file */
        char filename[256];
        snprintf(filename, sizeof(filename), "mpi_%dprocs_metrics.csv", size);
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
    }
    
    /* Cleanup */
    free_metrics(metrics);
    free_network(net);
    free_local_dataset(local_train);
    free_dataset(global_train);
    free_dataset(test_data);
    
    MPI_Finalize();
    return 0;
}
