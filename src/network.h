#ifndef NETWORK_H
#define NETWORK_H

/*
 * A small, fully connected feed-forward neural network.
 *
 * The network is stored explicitly as layers of neurons (rather than as
 * weight matrices) to keep the forward and backward pass easy to follow.
 * Every non-input neuron uses a sigmoid activation and the network is
 * trained with per-sample stochastic gradient descent on a mean squared
 * error loss.
 */

typedef struct {
    int num_inputs;       /* number of incoming connections (0 for input layer) */
    double *weights;      /* one weight per incoming connection                 */
    double bias;
    double weighted_sum;  /* z = w · a_prev + b                                 */
    double activation;    /* a = sigmoid(z)                                     */
    double delta;         /* dLoss/dz, filled in during backpropagation         */
} Neuron;

typedef struct {
    int num_neurons;
    Neuron *neurons;
} Layer;

typedef struct {
    int num_layers;       /* input + hidden + output */
    Layer *layers;
} Network;

/* Builds a network, e.g. sizes = {784, 64, 64, 10}. Weights ~ U(-0.5, 0.5). */
Network network_create(int num_layers, const int *layer_sizes);
void    network_free(Network *net);

/* Runs a forward pass; afterwards the output layer holds the predictions. */
void network_forward(Network *net, const double *input);

/* Backpropagates the error for one sample and applies an SGD update. */
void network_backward(Network *net, int target_label, double learning_rate);

/* Index of the output neuron with the highest activation (after a forward pass). */
int network_predict(const Network *net);

/* Squared error 0.5 * sum((target - output)^2) of the current output. */
double network_sample_loss(const Network *net, int target_label);

/*
 * Trains for `epochs` epochs. After every epoch the average training and
 * test loss is stored in train_loss[e] / test_loss[e] (either may be NULL).
 */
void network_train(Network *net,
                   double **train_x, const int *train_y, int train_n,
                   double **test_x, const int *test_y, int test_n,
                   int epochs, double learning_rate,
                   double *train_loss, double *test_loss);

/* Model persistence as a plain CSV file (architecture header + one row per neuron). */
int     network_save(const Network *net, const char *path);
Network network_load(const char *path);

#endif
