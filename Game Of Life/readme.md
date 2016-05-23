# Game of life parallel implementation using CUDA

Part of project series for the Parallel and Distributed Systems class.
For a really gangster approach to the problem I suggest you visit http://www.marekfiser.com/Projects/Conways-Game-of-Life-on-GPU-using-CUDA

Mine isn't optimal, but achieves significant speedups.

## gameOfLife_serial
This is a serial implementation of Conway's game of life, used as a benchmark to test performance improvements

## game

Conway's Game of Life CUDA implementation in C, using only global memory, with enough threads to process every cell in the grid.

## game2

Conway's Game of Life CUDA implementation in C, using only global memory, with grid stride loops.

Should expect same performance with game.cu, since NVIDIA says that grid stride loops have the same
instruction cost as monolithic kernels.

## game3_shared

Conway's Game of Life CUDA implementation in C, using shared memory.

Should see slight performance boost in devices with 1.x compute capability.

args are:
- filename
- N , NxN is the size of our initial grid ( a .bin file)
- t, threads per block
