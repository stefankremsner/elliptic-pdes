# Solve elliptic PDEs with Machine learning

## Run

For a single point run the following
```
python train.py [-h] [-E EXAMPLE] [-D DIM] [-N NUM_TIME_INTERVAL] [-T TOTAL_TIME] [-BS BATCH_SIZE] [-VS VALID_SIZE] [-I NUM_ITERATIONS]
```
For mulitple points run the parallelized version
```
python  train_parallel.py [-h] [-E EXAMPLE] [-D DIM] [-N NUM_TIME_INTERVAL] [-T TOTAL_TIME] [-BS BATCH_SIZE] [-VS VALID_SIZE] [-I NUM_ITERATIONS]
```