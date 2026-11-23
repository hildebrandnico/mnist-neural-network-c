/*
 * mnist-nn: train and evaluate a fully connected neural network on MNIST,
 * written from scratch in C (no ML libraries).
 */
#include <errno.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "dataset.h"
#include "metrics.h"
#include "network.h"

#define MAX_HIDDEN 16

typedef struct {
    const char *train_path;
    const char *test_path;
    int train_n;
    int test_n;
    int dim;
    int classes;
    int hidden[MAX_HIDDEN];
    int num_hidden;
    int epochs;
    double learning_rate;
    unsigned int seed;
    const char *out_dir;
    const char *load_path;
    int save_misclassified;
} Config;

static void usage(const char *prog) {
    printf(
        "Usage: %s [options]\n\n"
        "Data\n"
        "  --train FILE        training CSV            (default: data/mnist_train.csv)\n"
        "  --test FILE         test CSV                (default: data/mnist_test.csv)\n"
        "  --train-n N         max. training samples   (default: 60000)\n"
        "  --test-n N          max. test samples       (default: 10000)\n"
        "  --dim N             features per sample     (default: 784)\n"
        "  --classes N         number of classes       (default: 10)\n\n"
        "Model / training\n"
        "  --hidden A,B,...    hidden layer sizes      (default: 64,64)\n"
        "  --epochs N          training epochs         (default: 30)\n"
        "  --lr X              learning rate           (default: 0.1)\n"
        "  --seed N            RNG seed for weight init (default: current time)\n"
        "  --load FILE         skip training, evaluate a saved model\n\n"
        "Output\n"
        "  --out DIR           output directory        (default: output)\n"
        "  --save-misclassified  write misclassified test samples to DIR/misclassified.csv\n"
        "  -h, --help          show this help\n",
        prog);
}

static int parse_hidden(const char *arg, Config *cfg) {
    char buf[256];
    strncpy(buf, arg, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';

    cfg->num_hidden = 0;
    for (char *tok = strtok(buf, ","); tok != NULL; tok = strtok(NULL, ",")) {
        int size = atoi(tok);
        if (size <= 0 || cfg->num_hidden >= MAX_HIDDEN) return -1;
        cfg->hidden[cfg->num_hidden++] = size;
    }
    return cfg->num_hidden > 0 ? 0 : -1;
}

static void parse_args(int argc, char **argv, Config *cfg) {
    enum { OPT_TRAIN = 1000, OPT_TEST, OPT_TRAIN_N, OPT_TEST_N, OPT_DIM, OPT_CLASSES,
           OPT_HIDDEN, OPT_EPOCHS, OPT_LR, OPT_SEED, OPT_OUT, OPT_LOAD, OPT_MISCLASSIFIED };

    static const struct option options[] = {
        {"train", required_argument, NULL, OPT_TRAIN},
        {"test", required_argument, NULL, OPT_TEST},
        {"train-n", required_argument, NULL, OPT_TRAIN_N},
        {"test-n", required_argument, NULL, OPT_TEST_N},
        {"dim", required_argument, NULL, OPT_DIM},
        {"classes", required_argument, NULL, OPT_CLASSES},
        {"hidden", required_argument, NULL, OPT_HIDDEN},
        {"epochs", required_argument, NULL, OPT_EPOCHS},
        {"lr", required_argument, NULL, OPT_LR},
        {"seed", required_argument, NULL, OPT_SEED},
        {"out", required_argument, NULL, OPT_OUT},
        {"load", required_argument, NULL, OPT_LOAD},
        {"save-misclassified", no_argument, NULL, OPT_MISCLASSIFIED},
        {"help", no_argument, NULL, 'h'},
        {NULL, 0, NULL, 0},
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "h", options, NULL)) != -1) {
        switch (opt) {
            case OPT_TRAIN:   cfg->train_path = optarg; break;
            case OPT_TEST:    cfg->test_path = optarg; break;
            case OPT_TRAIN_N: cfg->train_n = atoi(optarg); break;
            case OPT_TEST_N:  cfg->test_n = atoi(optarg); break;
            case OPT_DIM:     cfg->dim = atoi(optarg); break;
            case OPT_CLASSES: cfg->classes = atoi(optarg); break;
            case OPT_EPOCHS:  cfg->epochs = atoi(optarg); break;
            case OPT_LR:      cfg->learning_rate = atof(optarg); break;
            case OPT_SEED:    cfg->seed = (unsigned int)strtoul(optarg, NULL, 10); break;
            case OPT_OUT:     cfg->out_dir = optarg; break;
            case OPT_LOAD:    cfg->load_path = optarg; break;
            case OPT_MISCLASSIFIED: cfg->save_misclassified = 1; break;
            case OPT_HIDDEN:
                if (parse_hidden(optarg, cfg) != 0) {
                    fprintf(stderr, "error: invalid --hidden '%s' (expected e.g. 64,64)\n", optarg);
                    exit(EXIT_FAILURE);
                }
                break;
            case 'h': usage(argv[0]); exit(EXIT_SUCCESS);
            default:  usage(argv[0]); exit(EXIT_FAILURE);
        }
    }

    if (cfg->train_n <= 0 || cfg->test_n <= 0 || cfg->dim <= 0 || cfg->classes <= 1 ||
        cfg->epochs <= 0 || cfg->learning_rate <= 0.0) {
        fprintf(stderr, "error: sample counts, dim, epochs and lr must be positive, classes > 1\n");
        exit(EXIT_FAILURE);
    }
}

static void ensure_dir(const char *path) {
    if (mkdir(path, 0755) != 0 && errno != EEXIST) {
        fprintf(stderr, "error: cannot create directory '%s'\n", path);
        exit(EXIT_FAILURE);
    }
}

static FILE *open_in(const char *dir, const char *name, const char *mode) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    FILE *f = fopen(path, mode);
    if (f == NULL) {
        fprintf(stderr, "error: cannot open '%s'\n", path);
        exit(EXIT_FAILURE);
    }
    return f;
}

static void write_loss(const char *dir, const double *train_loss, const double *test_loss, int epochs) {
    FILE *f = open_in(dir, "loss.csv", "w");
    fprintf(f, "epoch,train_loss,test_loss\n");
    for (int e = 0; e < epochs; e++) {
        fprintf(f, "%d,%.8f,%.8f\n", e + 1, train_loss[e], test_loss[e]);
    }
    fclose(f);
}

/* Appends one row to DIR/experiments.csv so many runs can be compared later. */
static void append_experiment(const char *dir, const Config *cfg, int train_n, int test_n,
                              double sec_per_epoch, const Metrics *m) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/experiments.csv", dir);
    FILE *probe = fopen(path, "r");
    int is_new = (probe == NULL);
    if (probe) fclose(probe);

    FILE *f = open_in(dir, "experiments.csv", "a");
    if (is_new) {
        fprintf(f, "hidden_layers,train_samples,test_samples,epochs,learning_rate,seed,"
                   "sec_per_epoch,accuracy,macro_precision,macro_recall,macro_f1\n");
    }
    for (int i = 0; i < cfg->num_hidden; i++) {
        fprintf(f, "%s%d", i ? "-" : "", cfg->hidden[i]);
    }
    fprintf(f, ",%d,%d,%d,%g,%u,%.6f,%.2f,%.2f,%.2f,%.2f\n",
            train_n, test_n, cfg->epochs, cfg->learning_rate, cfg->seed,
            sec_per_epoch, m->accuracy, m->macro_precision, m->macro_recall, m->macro_f1);
    fclose(f);
}

int main(int argc, char **argv) {
    Config cfg = {
        .train_path = "data/mnist_train.csv",
        .test_path = "data/mnist_test.csv",
        .train_n = 60000,
        .test_n = 10000,
        .dim = 784,
        .classes = 10,
        .hidden = {64, 64},
        .num_hidden = 2,
        .epochs = 30,
        .learning_rate = 0.1,
        .seed = (unsigned int)time(NULL),
        .out_dir = "output",
        .load_path = NULL,
        .save_misclassified = 0,
    };
    parse_args(argc, argv, &cfg);
    srand(cfg.seed);
    ensure_dir(cfg.out_dir);

    printf("Loading test data from '%s'...\n", cfg.test_path);
    Dataset test = dataset_load_csv(cfg.test_path, cfg.test_n, cfg.dim);
    printf("  %d samples\n", test.n);

    Network net;
    double sec_per_epoch = 0.0;
    int train_n = 0;

    if (cfg.load_path != NULL) {
        printf("Loading model from '%s'...\n", cfg.load_path);
        net = network_load(cfg.load_path);

        int model_inputs = net.layers[0].num_neurons;
        int model_classes = net.layers[net.num_layers - 1].num_neurons;
        if (model_inputs != cfg.dim || model_classes != cfg.classes) {
            fprintf(stderr, "error: model expects %d inputs / %d classes, but --dim is %d and --classes is %d\n",
                    model_inputs, model_classes, cfg.dim, cfg.classes);
            exit(EXIT_FAILURE);
        }
    } else {
        printf("Loading training data from '%s'...\n", cfg.train_path);
        Dataset train = dataset_load_csv(cfg.train_path, cfg.train_n, cfg.dim);
        train_n = train.n;
        if (train.n == 0) {
            fprintf(stderr, "error: no training samples could be read from '%s'\n", cfg.train_path);
            exit(EXIT_FAILURE);
        }
        printf("  %d samples\n", train.n);

        int sizes[MAX_HIDDEN + 2];
        int num_layers = 0;
        sizes[num_layers++] = cfg.dim;
        for (int i = 0; i < cfg.num_hidden; i++) sizes[num_layers++] = cfg.hidden[i];
        sizes[num_layers++] = cfg.classes;

        printf("\nArchitecture: ");
        for (int i = 0; i < num_layers; i++) printf("%s%d", i ? " -> " : "", sizes[i]);
        printf("   (epochs: %d, learning rate: %g, seed: %u)\n\n", cfg.epochs, cfg.learning_rate, cfg.seed);

        net = network_create(num_layers, sizes);

        double *train_loss = malloc((size_t)cfg.epochs * sizeof(double));
        double *test_loss = malloc((size_t)cfg.epochs * sizeof(double));

        clock_t start = clock();
        network_train(&net, train.x, train.y, train.n, test.x, test.y, test.n,
                      cfg.epochs, cfg.learning_rate, train_loss, test_loss);
        sec_per_epoch = ((double)(clock() - start) / CLOCKS_PER_SEC) / cfg.epochs;
        printf("Training time: %.3f s per epoch\n", sec_per_epoch);

        write_loss(cfg.out_dir, train_loss, test_loss, cfg.epochs);
        free(train_loss);
        free(test_loss);
        dataset_free(&train);

        char model_path[1024];
        snprintf(model_path, sizeof(model_path), "%s/model.csv", cfg.out_dir);
        if (network_save(&net, model_path) == 0) {
            printf("Model saved to '%s'\n", model_path);
        }
    }

    FILE *miss = NULL;
    if (cfg.save_misclassified) {
        miss = open_in(cfg.out_dir, "misclassified.csv", "w");
        for (int j = 0; j < cfg.dim; j++) fprintf(miss, "x%d,", j + 1);
        fprintf(miss, "actual,predicted\n");
    }

    Metrics m = metrics_evaluate(&net, &test, miss);
    if (miss) fclose(miss);
    metrics_print(&m);

    if (cfg.load_path == NULL) {
        append_experiment(cfg.out_dir, &cfg, train_n, test.n, sec_per_epoch, &m);
    }

    metrics_free(&m);
    network_free(&net);
    dataset_free(&test);
    return EXIT_SUCCESS;
}
