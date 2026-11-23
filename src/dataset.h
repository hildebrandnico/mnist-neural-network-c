#ifndef DATASET_H
#define DATASET_H

/*
 * A labelled dataset loaded from CSV.
 *
 * Expected format: one header line, then one sample per line with `dim`
 * pixel values (0-255) followed by the integer class label, e.g.
 *     x1,x2,...,x784,y
 *     0,0,...,0,7
 * Pixel values are scaled to [0, 1] while loading.
 */
typedef struct {
    int n;          /* number of samples actually loaded */
    int dim;        /* features per sample               */
    double **x;     /* n x dim                            */
    int *y;         /* n labels                           */
} Dataset;

/* Loads at most `max_samples` rows. Exits with an error if the file is missing. */
Dataset dataset_load_csv(const char *path, int max_samples, int dim);
void    dataset_free(Dataset *ds);

#endif
