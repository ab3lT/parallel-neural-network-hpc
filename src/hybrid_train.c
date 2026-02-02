/**
 * hybrid_train.c - Hybrid MPI+OpenMP parallel implementation
 * 
 * Parallelization of Deep Learning Models
 * Distributed Computing for AI - Masters Assignment
 * 
 * Parallelization Strategy: Hierarchical Data Parallelism
 *   - MPI: Data partitioned across compute nodes (distributed memory)
 *   - OpenMP: Parallel processing within each node (shared memory)
 *   - Two-level gradient aggregation:
 *     1. OpenMP threads aggregate locally within a process
 *     2. MPI processes aggregate globally across nodes
 * 
 * Target Architecture: Hybrid cluster with multicore nodes
 * 
 * Communication Pattern:
 *   - Inter-node: MPI_Allreduce for gradient averaging
 *   - Intra-node: OpenMP critical sections for thread-local gradient accumulation
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
#include <omp.h>
#include "neural_net.h"

/*============================================================================
 * Local dataset structure (same as MPI version)
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
    
    int base_samples = global_data->num_samples / size;
    int remainder = global_data->num_samples % size;
    
    int start_idx;
    if (rank < remainder) {
        local->local_num_samples = base_samples + 1;
        start_idx = rank * (base_samples + 1);
    } else {
        local->local_num_samples = base_samples;
        start_idx = remainder * (base_samples + 1) + (rank - remainder) * base_samples;
    }
    
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
        
        for (int k = 0; k < input_size; k++) {
            double temp = local->local_data[i * input_size + k];
            local->local_data[i * input_size + k] = local->local_data[j * input_size + k];
            local->local_data[j * input_size + k] = temp;
        }
        
        int temp_label = local->local_labels[i];
        local->local_labels[i] = local->local_labels[j];
        local->local_labels[j] = temp_label;
    }
}

/*============================================================================
 * Thread-local network (same as OpenMP version)
 *============================================================================*/

typedef struct {
    NeuralNetwork *net;
    double *grad_weights_accum;
    double *grad_biases_accum;
    int total_grad_weights_size;
    int total_grad_biases_size;
} ThreadLocalNetwork;

ThreadLocalNetwork* create_thread_local_network(NeuralNetwork *base_net) {
    ThreadLocalNetwork *tln = (ThreadLocalNetwork*)malloc(sizeof(ThreadLocalNetwork));
    
    tln->net = create_network(base_net->learning_rate);
    copy_network(tln->net, base_net);
    
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
 * Training function - Hybrid MPI+OpenMP Implementation
 *============================================================================*/

TrainingMetrics* train_hybrid(NeuralNetwork *net, LocalDataset *local_train,
                              Dataset *test_data, int num_epochs,
                              int batch_size, int rank, int size,
                              int num_threads) {
    TrainingMetrics *metrics = create_metrics(num_epochs);
    
    omp_set_num_threads(num_threads);
    
    /* Calculate total gradient sizes */
    int total_grad_w_size = 0, total_grad_b_size = 0;
    for (int l = 0; l < net->num_layers; l++) {
        total_grad_w_size += net->layers[l].input_size * net->layers[l].output_size;
        total_grad_b_size += net->layers[l].output_size;
    }
    int total_grad_size = total_grad_w_size + total_grad_b_size;
    
    /* Allocate gradient buffers */
    double *process_grads = (double*)calloc(total_grad_size, sizeof(double));
    double *global_grads = (double*)calloc(total_grad_size, sizeof(double));
    
    int local_num_batches = (local_train->local_num_samples + batch_size - 1) / batch_size;
    
    if (rank == 0) {
        printf("\n=== Hybrid MPI+OpenMP Parallel Training ===\n");
        printf("MPI processes: %d\n", size);
        printf("OpenMP threads per process: %d\n", num_threads);
        printf("Total parallelism: %d\n", size * num_threads);
        printf("Total training samples: %d\n", local_train->global_num_samples);
        printf("Local samples (rank 0): %d\n", local_train->local_num_samples);
        printf("Test samples: %d\n", test_data->num_samples);
        printf("Batch size: %d\n", batch_size);
        printf("Local batches per epoch: %d\n", local_num_batches);
        printf("Epochs: %d\n\n", num_epochs);
    }
    
    /* Create thread-local networks */
    ThreadLocalNetwork **thread_nets = (ThreadLocalNetwork**)malloc(
        num_threads * sizeof(ThreadLocalNetwork*));
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        thread_nets[tid] = create_thread_local_network(net);
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
            
            /* Reset process-level gradients */
            memset(process_grads, 0, total_grad_size * sizeof(double));
            
            double batch_loss = 0.0;
            int batch_correct = 0;
            
            /* OpenMP parallel processing within MPI process */
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
                    double *input = &local_train->local_data[sample_idx * INPUT_SIZE];
                    int label = local_train->local_labels[sample_idx];
                    
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
                
                /* Accumulate thread-local gradients to process-level */
                #pragma omp critical
                {
                    for (int i = 0; i < total_grad_w_size; i++) {
                        process_grads[i] += tln->grad_weights_accum[i];
                    }
                    for (int i = 0; i < total_grad_b_size; i++) {
                        process_grads[total_grad_w_size + i] += tln->grad_biases_accum[i];
                    }
                }
            }
            
            local_epoch_loss += batch_loss;
            local_correct += batch_correct;
            
            /* MPI Allreduce to aggregate gradients across processes */
            MPI_Allreduce(process_grads, global_grads, total_grad_size,
                         MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            
            /* Get total batch size across all processes */
            int total_batch_size;
            MPI_Allreduce(&actual_batch_size, &total_batch_size, 1,
                         MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            
            /* Update weights with globally averaged gradients */
            int offset = 0;
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
    
    /* Cleanup thread-local networks */
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        free_thread_local_network(thread_nets[tid]);
    }
    free(thread_nets);
    free(process_grads);
    free(global_grads);
    
    return metrics;
}

/*============================================================================
 * Main function
 *============================================================================*/

int main(int argc, char *argv[]) {
    int rank, size;
    int provided;
    
    /* Initialize MPI with thread support */
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        printf("Warning: MPI does not support required threading level\n");
    }
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
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
        } else if ((strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) && rank == 0) {
            printf("Usage: mpirun -np <procs> %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  -n <samples>        Number of training samples (default: 10000)\n");
            printf("  -e <epochs>         Number of epochs (default: 20)\n");
            printf("  -b <batch>          Batch size per process (default: 64)\n");
            printf("  -lr <rate>          Learning rate (default: 0.01)\n");
            printf("  -t <threads>        OpenMP threads per process (default: auto)\n");
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
        printf("  Hybrid MPI+OpenMP Neural Network Training\n");
        printf("=================================================\n");
        printf("\nConfiguration:\n");
        printf("  Network Architecture: %d -> %d -> %d -> %d\n", 
               INPUT_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE, OUTPUT_SIZE);
        printf("  MPI processes: %d\n", size);
        printf("  OpenMP threads per process: %d\n", num_threads);
        printf("  Total parallelism: %d\n", size * num_threads);
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
    
    /* Create and initialize network */
    if (rank == 0) printf("Initializing neural network...\n");
    NeuralNetwork *net = create_network(learning_rate);
    initialize_weights(net, 42);
    
    /* Synchronize initial weights */
    MPI_Barrier(MPI_COMM_WORLD);
    
    /* Train the network */
    TrainingMetrics *metrics = train_hybrid(net, local_train, test_data, 
                                            num_epochs, batch_size, 
                                            rank, size, num_threads);
    
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
        snprintf(filename, sizeof(filename), "hybrid_%dprocs_%dthreads_metrics.csv", 
                 size, num_threads);
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
