#include "dataset.h"

#include <stdio.h>
#include <stdlib.h>

Dataset dataset_load_csv(const char *path, int max_samples, int dim) {
    FILE *f = fopen(path, "r");
    if (f == NULL) {
        fprintf(stderr, "error: cannot open '%s' (run `make data` to download MNIST)\n", path);
        exit(EXIT_FAILURE);
    }

    Dataset ds;
    ds.n = 0;
    ds.dim = dim;
    ds.x = malloc((size_t)max_samples * sizeof(double *));
    ds.y = malloc((size_t)max_samples * sizeof(int));
    if (ds.x == NULL || ds.y == NULL) {
        fprintf(stderr, "error: out of memory\n");
        exit(EXIT_FAILURE);
    }

    /* Skip the header line. */
    int c;
    while ((c = fgetc(f)) != EOF && c != '\n') {}

    for (int i = 0; i < max_samples; i++) {
        double *row = malloc((size_t)dim * sizeof(double));
        if (row == NULL) {
            fprintf(stderr, "error: out of memory\n");
            exit(EXIT_FAILURE);
        }
        int ok = 1;
        for (int j = 0; j < dim && ok; j++) {
            if (fscanf(f, "%lf,", &row[j]) != 1) ok = 0;
            else row[j] /= 255.0;  /* normalise pixels to [0, 1] */
        }
        if (!ok || fscanf(f, "%d", &ds.y[i]) != 1) {
            free(row);
            break;
        }
        ds.x[i] = row;
        ds.n++;
    }

    fclose(f);
    return ds;
}

void dataset_free(Dataset *ds) {
    for (int i = 0; i < ds->n; i++) {
        free(ds->x[i]);
    }
    free(ds->x);
    free(ds->y);
    ds->x = NULL;
    ds->y = NULL;
    ds->n = 0;
}
