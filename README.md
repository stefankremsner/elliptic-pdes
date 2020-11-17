# Solve elliptic PDEs with Machine learning

examples include

```
NonequidistantTimestepsInsurance
NonequidistantLaplaceOnSmallerBall
NonequidistantQuadraticZ
```

can be set in .env file for a default value

## Run

For a single point run the following
```
python train.py [-h] [-E EXAMPLE] [-D DIM] [-N NUM_TIME_INTERVAL] [-T TOTAL_TIME] [-BS BATCH_SIZE] [-VS validation_size] [-I NUM_ITERATIONS]
```
For mulitple points run the parallelized version
```
python  train_parallel.py [-h] [-E EXAMPLE] [-D DIM] [-N NUM_TIME_INTERVAL] [-T TOTAL_TIME] [-BS BATCH_SIZE] [-VS validation_size] [-I NUM_ITERATIONS]
```

the code is based on https://github.com/frankhan91/DeepBSDE
