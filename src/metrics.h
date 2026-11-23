#ifndef METRICS_H
#define METRICS_H

#include <stdio.h>

#include "dataset.h"
#include "network.h"

/* Evaluation results for a multi-class classifier. */
typedef struct {
    int num_classes;
    int **confusion;        /* confusion[actual][predicted]        */
    int correct;
    int total;
    double accuracy;        /* in percent                          */
    double *precision;      /* per class, in percent               */
    double *recall;         /* per class, in percent               */
    double *f1;             /* per class, in percent               */
    double macro_precision; /* unweighted mean over all classes    */
    double macro_recall;
    double macro_f1;
} Metrics;

/*
 * Runs the network on every sample of `ds` and computes the confusion matrix
 * and per-class metrics. If `misclassified` is non-NULL, every wrongly
 * classified sample is appended to it (raw pixels + predicted label).
 */
Metrics metrics_evaluate(Network *net, const Dataset *ds, FILE *misclassified);

void metrics_print(const Metrics *m);
void metrics_free(Metrics *m);

#endif
