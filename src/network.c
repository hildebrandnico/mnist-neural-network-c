#include "network.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------------ */
/* Construction / destruction                                                */
/* ------------------------------------------------------------------------ */

static void *xmalloc(size_t size) {
    void *p = malloc(size);
    if (p == NULL) {
        fprintf(stderr, "error: out of memory\n");
        exit(EXIT_FAILURE);
    }
    return p;
}

static Neuron neuron_create(int num_inputs) {
    Neuron n;
    n.num_inputs = num_inputs;
    n.bias = 0.0;
    n.weighted_sum = 0.0;
    n.activation = 0.0;
    n.delta = 0.0;
    n.weights = NULL;  /* input-layer neurons have no incoming weights */

    if (num_inputs > 0) {
        n.weights = xmalloc((size_t)num_inputs * sizeof(double));
        for (int i = 0; i < num_inputs; i++) {
            n.weights[i] = ((double)rand() / (double)RAND_MAX) - 0.5;
        }
    }
    return n;
}

static Layer layer_create(int num_neurons, int inputs_per_neuron) {
    Layer l;
    l.num_neurons = num_neurons;
    l.neurons = xmalloc((size_t)num_neurons * sizeof(Neuron));
    for (int i = 0; i < num_neurons; i++) {
        l.neurons[i] = neuron_create(inputs_per_neuron);
    }
    return l;
}

Network network_create(int num_layers, const int *layer_sizes) {
    Network net;
    net.num_layers = num_layers;
    net.layers = xmalloc((size_t)num_layers * sizeof(Layer));
    for (int i = 0; i < num_layers; i++) {
        int inputs = (i == 0) ? 0 : layer_sizes[i - 1];
        net.layers[i] = layer_create(layer_sizes[i], inputs);
    }
    return net;
}

void network_free(Network *net) {
    for (int i = 0; i < net->num_layers; i++) {
        for (int j = 0; j < net->layers[i].num_neurons; j++) {
            free(net->layers[i].neurons[j].weights);
        }
        free(net->layers[i].neurons);
    }
    free(net->layers);
    net->layers = NULL;
    net->num_layers = 0;
}

/* ------------------------------------------------------------------------ */
/* Math                                                                      */
/* ------------------------------------------------------------------------ */

static double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

/* Derivative of the sigmoid, expressed through its output a = sigmoid(z). */
static double sigmoid_derivative(double a) {
    return a * (1.0 - a);
}

/* ------------------------------------------------------------------------ */
/* Forward / backward pass                                                   */
/* ------------------------------------------------------------------------ */

void network_forward(Network *net, const double *input) {
    Layer *in = &net->layers[0];
    for (int i = 0; i < in->num_neurons; i++) {
        in->neurons[i].activation = input[i];
    }

    for (int l = 1; l < net->num_layers; l++) {
        const Layer *prev = &net->layers[l - 1];
        Layer *curr = &net->layers[l];

        for (int j = 0; j < curr->num_neurons; j++) {
            Neuron *n = &curr->neurons[j];
            double z = n->bias;
            for (int k = 0; k < prev->num_neurons; k++) {
                z += prev->neurons[k].activation * n->weights[k];
            }
            n->weighted_sum = z;
            n->activation = sigmoid(z);
        }
    }
}

void network_backward(Network *net, int target_label, double learning_rate) {
    int last = net->num_layers - 1;

    /* 1. Output layer: delta = (a - t) * sigmoid'(z)  (gradient of MSE) */
    Layer *out = &net->layers[last];
    for (int i = 0; i < out->num_neurons; i++) {
        double a = out->neurons[i].activation;
        double t = (i == target_label) ? 1.0 : 0.0;
        out->neurons[i].delta = (a - t) * sigmoid_derivative(a);
    }

    /* 2. Hidden layers: propagate the error backwards through the weights */
    for (int l = last - 1; l > 0; l--) {
        Layer *curr = &net->layers[l];
        const Layer *next = &net->layers[l + 1];

        for (int j = 0; j < curr->num_neurons; j++) {
            double err = 0.0;
            for (int k = 0; k < next->num_neurons; k++) {
                err += next->neurons[k].delta * next->neurons[k].weights[j];
            }
            curr->neurons[j].delta = err * sigmoid_derivative(curr->neurons[j].activation);
        }
    }

    /* 3. Gradient descent step on all weights and biases */
    for (int l = 1; l < net->num_layers; l++) {
        Layer *curr = &net->layers[l];
        const Layer *prev = &net->layers[l - 1];

        for (int j = 0; j < curr->num_neurons; j++) {
            Neuron *n = &curr->neurons[j];
            double step = learning_rate * n->delta;
            n->bias -= step;
            for (int k = 0; k < prev->num_neurons; k++) {
                n->weights[k] -= step * prev->neurons[k].activation;
            }
        }
    }
}

int network_predict(const Network *net) {
    const Layer *out = &net->layers[net->num_layers - 1];
    int best = 0;
    for (int k = 1; k < out->num_neurons; k++) {
        if (out->neurons[k].activation > out->neurons[best].activation) {
            best = k;
        }
    }
    return best;
}

double network_sample_loss(const Network *net, int target_label) {
    const Layer *out = &net->layers[net->num_layers - 1];
    double loss = 0.0;
    for (int j = 0; j < out->num_neurons; j++) {
        double t = (j == target_label) ? 1.0 : 0.0;
        double diff = t - out->neurons[j].activation;
        loss += 0.5 * diff * diff;
    }
    return loss;
}

/* ------------------------------------------------------------------------ */
/* Training loop                                                             */
/* ------------------------------------------------------------------------ */

void network_train(Network *net,
                   double **train_x, const int *train_y, int train_n,
                   double **test_x, const int *test_y, int test_n,
                   int epochs, double learning_rate,
                   double *train_loss, double *test_loss) {
    for (int e = 0; e < epochs; e++) {
        /* Test loss of the current model (before this epoch's updates). */
        double epoch_test_loss = 0.0;
        for (int i = 0; i < test_n; i++) {
            network_forward(net, test_x[i]);
            epoch_test_loss += network_sample_loss(net, test_y[i]);
        }

        /* One pass of per-sample SGD over the training set. */
        double epoch_train_loss = 0.0;
        for (int i = 0; i < train_n; i++) {
            network_forward(net, train_x[i]);
            epoch_train_loss += network_sample_loss(net, train_y[i]);
            network_backward(net, train_y[i], learning_rate);
        }

        double avg_train = epoch_train_loss / train_n;
        double avg_test = (test_n > 0) ? epoch_test_loss / test_n : 0.0;
        if (train_loss) train_loss[e] = avg_train;
        if (test_loss)  test_loss[e] = avg_test;

        printf("\rEpoch %4d / %d   train loss: %.6f   test loss: %.6f",
               e + 1, epochs, avg_train, avg_test);
        fflush(stdout);
    }
    printf("\n");
}

/* ------------------------------------------------------------------------ */
/* Persistence                                                               */
/* ------------------------------------------------------------------------ */

int network_save(const Network *net, const char *path) {
    FILE *f = fopen(path, "w");
    if (f == NULL) {
        fprintf(stderr, "error: cannot write model to '%s'\n", path);
        return -1;
    }

    fprintf(f, "ARCH");
    for (int i = 0; i < net->num_layers; i++) {
        fprintf(f, ",%d", net->layers[i].num_neurons);
    }
    fprintf(f, "\n");

    /* One row per neuron: layer, index, bias, weights... */
    for (int l = 1; l < net->num_layers; l++) {
        const Layer *layer = &net->layers[l];
        for (int j = 0; j < layer->num_neurons; j++) {
            const Neuron *n = &layer->neurons[j];
            fprintf(f, "%d,%d,%.15f", l, j, n->bias);
            for (int k = 0; k < n->num_inputs; k++) {
                fprintf(f, ",%.15f", n->weights[k]);
            }
            fprintf(f, "\n");
        }
    }
    fclose(f);
    return 0;
}

#define MAX_LAYERS 64

Network network_load(const char *path) {
    FILE *f = fopen(path, "r");
    if (f == NULL) {
        fprintf(stderr, "error: cannot open model file '%s'\n", path);
        exit(EXIT_FAILURE);
    }

    /* A row holds up to ~1000 weights of ~20 characters each. */
    size_t line_cap = 1 << 20;
    char *line = xmalloc(line_cap);

    if (fgets(line, (int)line_cap, f) == NULL || strncmp(line, "ARCH", 4) != 0) {
        fprintf(stderr, "error: '%s' is not a valid model file\n", path);
        exit(EXIT_FAILURE);
    }

    int sizes[MAX_LAYERS];
    int num_layers = 0;
    strtok(line, ",");  /* skip "ARCH" */
    for (char *tok = strtok(NULL, ","); tok != NULL && num_layers < MAX_LAYERS; tok = strtok(NULL, ",")) {
        sizes[num_layers++] = atoi(tok);
    }

    Network net = network_create(num_layers, sizes);

    while (fgets(line, (int)line_cap, f) != NULL) {
        char *tok = strtok(line, ",");
        if (tok == NULL) continue;
        int l = atoi(tok);
        if ((tok = strtok(NULL, ",")) == NULL) continue;
        int j = atoi(tok);
        if (l <= 0 || l >= num_layers || j < 0 || j >= net.layers[l].num_neurons) {
            fprintf(stderr, "error: invalid neuron index %d/%d in '%s'\n", l, j, path);
            exit(EXIT_FAILURE);
        }
        Neuron *n = &net.layers[l].neurons[j];
        if ((tok = strtok(NULL, ",")) == NULL) continue;
        n->bias = atof(tok);
        for (int k = 0; k < n->num_inputs; k++) {
            if ((tok = strtok(NULL, ",")) == NULL) {
                fprintf(stderr, "error: truncated row for layer %d neuron %d\n", l, j);
                exit(EXIT_FAILURE);
            }
            n->weights[k] = atof(tok);
        }
    }

    free(line);
    fclose(f);
    return net;
}
