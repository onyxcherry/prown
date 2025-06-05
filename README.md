# Projekt – Przetwarzanie Równoległe

## Build

### Potrzebne pliki

    - https://github.com/lvandeve/lodepng/blob/master/lodepng.h
    - https://github.com/lvandeve/lodepng/blob/master/lodepng.cpp

#### wersja sekwencyjna

```shell
g++ main.cpp lodepng.cpp -o program -O3 -Wall -Wextra
```

#### wersja z OpenMP

```shell
g++ main.cpp lodepng.cpp -o program_parallel -O3 -Wall -Wextra -fopenmp
```

#### wersja CUDA

```shell
nvcc -O3 main.cu -o main_cuda
```

## Uruchomienie

### Sekwencyjny/OpenMP

```shell
./main measure 6 20000000 165888000
```

### CUDA

```shell
./main 6 20000000 165888000
```
