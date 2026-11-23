#include "metrics.h"

#include <stdlib.h>

Metrics metrics_evaluate(Network *net, const Dataset *ds, FILE *misclassified) {
    Metrics m;
    int k = net->layers[net->num_layers - 1].num_neurons;

    m.num_classes = k;
    m.correct = 0;
    m.total = ds->n;
    m.confusion = malloc((size_t)k * sizeof(int *));
    for (int i = 0; i < k; i++) {
        m.confusion[i] = calloc((size_t)k, sizeof(int));
    }
    m.precision = calloc((size_t)k, sizeof(double));
    m.recall = calloc((size_t)k, sizeof(double));
    m.f1 = calloc((size_t)k, sizeof(double));

    /* Confusion matrix */
    for (int i = 0; i < ds->n; i++) {
        network_forward(net, ds->x[i]);
        int predicted = network_predict(net);
        int actual = ds->y[i];

        if (predicted == actual) {
            m.correct++;
        } else if (misclassified != NULL) {
            for (int j = 0; j < ds->dim; j++) {
                fprintf(misclassified, "%d,", (int)(ds->x[i][j] * 255.0 + 0.5));
            }
            fprintf(misclassified, "%d,%d\n", actual, predicted);
        }
        if (actual >= 0 && actual < k) {
            m.confusion[actual][predicted]++;
        }
    }
    m.accuracy = (m.total > 0) ? 100.0 * m.correct / m.total : 0.0;

    /* Per-class precision / recall / F1 and their macro averages */
    m.macro_precision = m.macro_recall = m.macro_f1 = 0.0;
    for (int c = 0; c < k; c++) {
        double tp = m.confusion[c][c];
        double actual_c = 0.0, predicted_c = 0.0;
        for (int j = 0; j < k; j++) {
            actual_c += m.confusion[c][j];
            predicted_c += m.confusion[j][c];
        }
        m.recall[c] = (actual_c > 0) ? 100.0 * tp / actual_c : 0.0;
        m.precision[c] = (predicted_c > 0) ? 100.0 * tp / predicted_c : 0.0;
        double pr = m.precision[c] + m.recall[c];
        m.f1[c] = (pr > 0) ? 2.0 * m.precision[c] * m.recall[c] / pr : 0.0;

        m.macro_precision += m.precision[c] / k;
        m.macro_recall += m.recall[c] / k;
        m.macro_f1 += m.f1[c] / k;
    }
    return m;
}

void metrics_print(const Metrics *m) {
    int k = m->num_classes;

    printf("\nAccuracy: %.2f%%  (%d / %d correct)\n\n", m->accuracy, m->correct, m->total);
    printf("Confusion matrix (rows = actual, columns = predicted):\n\n");

    printf("      ");
    for (int c = 0; c < k; c++) printf("%6d", c);
    printf("  | recall     F1\n");
    printf("------");
    for (int c = 0; c < k; c++) printf("------");
    printf("--+--------------\n");

    for (int r = 0; r < k; r++) {
        printf("%4d  ", r);
        for (int c = 0; c < k; c++) printf("%6d", m->confusion[r][c]);
        printf("  | %6.2f  %6.2f\n", m->recall[r], m->f1[r]);
    }

    printf("prec  ");
    for (int c = 0; c < k; c++) printf("%6.1f", m->precision[c]);
    printf("\n\n");

    printf("Macro precision: %.2f%%   macro recall: %.2f%%   macro F1: %.2f%%\n",
           m->macro_precision, m->macro_recall, m->macro_f1);
}

void metrics_free(Metrics *m) {
    for (int i = 0; i < m->num_classes; i++) {
        free(m->confusion[i]);
    }
    free(m->confusion);
    free(m->precision);
    free(m->recall);
    free(m->f1);
}
