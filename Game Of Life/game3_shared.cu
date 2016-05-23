/*

Conway's Game of Life CUDA implementation in C, using shared memory.

Should see slight performance boost in devices with 1.x compute capability.

Papavasileiou Themis 24/01/2015 

*/


#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <time.h>
#include <sys/time.h>
#include <stdint.h>

//Length of our 1-dimensional block.
#define BLOCKLEN 1024

void die(const char *message){
    if(errno){
        perror(message);
    }else{
        printf("Error: %s\n", message);
    }
    exit(1);
}

void warn(const char *message){
    if(errno){
        perror(message);
    }else{
        printf("Warning: %s\n", message);
    }
    return;
}


void read_from_file(int *X, char *filename, int N){

    FILE *fp = fopen(filename, "r+");
    int size = fread(X, sizeof(int), N*N, fp);
    if(!fp)
        die("Couldn't open file to read");
    if(!size)
        die("Couldn't read from file");
    if(N*N != size)
        warn("Expected to read different number of elements");

    printf("elements read: %d\n", size);

    fclose(fp);
    return;
}

void write_to_file(int *X, char *filename, int N){

    //save as tableNxN_new.bin
    char newfilename[100];
    sprintf(newfilename, "cuda_table%dx%d.bin", N, N);
    printf("writing to: %s\n", newfilename);

    FILE *fp;
    int size;
    if( ! ( fp = fopen(newfilename, "w+") ) )
        die("Couldn't open file to write");
    if( ! (size = fwrite(X, sizeof(int), N*N, fp)) )
        die("Couldn't write to file");
    if (size != N*N)
        warn("Expected to write different number of elements");

    fclose(fp);
    return;
}


void printCells(int *table, int N){
    int j=0, i=0;
    for(i=0; i<10; i++){
        for(j=0; j<10; j++){
            printf("%d ", table[N*i+j]);
        }
        printf("\n");
    }
}


/*
    
    In order to attend to the cyclic boundaries every block needs to hold the values of its neighbors in the shared memory.
    So the shared memory array needs to be 3*N long, holding 3 lines. The one above, the one the cells we're interested in are in, and the one below.
    We'll keep using the i*N+j notation so the layout of the shared memory array will be 
        > row 0 has the above line
        > row 1 has own line
        > row 2 has below line.
*/
__global__ void evolve(int *X, int *new_table, int N){

    //Every block works in a row.
    int global_i = blockIdx.x; //index of the row
    int temp_i = global_i; 

    //shared memory array 3*N long
    extern __shared__ int shared_memory[];

    //read row above (i-1) and write to shared_memory[0][j]
    if (global_i == 0){
        //cyclic boundary condition
        temp_i = N - 1; 
    }else{
        //else it's business as usual, we look one row up.
        temp_i = global_i - 1; 
    }
    //block-strides to fill the shared memory
    for(int j = threadIdx.x; j < N; j+=blockDim.x){
        if(j<N){
            shared_memory[0*N + j] = X[(temp_i)*N + j];
        }
    }

    //same idea for own row.
    for(int j = threadIdx.x; j < N; j+=blockDim.x){
            shared_memory[1*N + j] = X[(global_i)*N + j];
    }

    //same for row below
    if(global_i == N-1){
        temp_i = 0; 
    }else{
        temp_i = global_i + 1;
    }
    for(int j = threadIdx.x; j < N; j+=blockDim.x){
            shared_memory[2*N + j] = X[(temp_i)*N + j];
    }

    //sync before any writing in the shared memory takes place!
    __syncthreads();

    //Let's write!
    //Notice that we abandoned the if then else statement for readability this time.
    //Most of the implementations online seem to be using this form.
    int left, right;
    //Each thread starts at its own point and continues by block-strides, while it's in bounds of course.
    for(int j = threadIdx.x; j < N; j+=blockDim.x){
        // Marek Fiser's blog says __mul24 saves cycles. His implementation is really cool, I trust him.
        int idx = __mul24(N, global_i) + j;

        //tend to the cyclic boundary conditions
        left =  j == 0 ? N - 1 : j - 1;
        right = j == N-1 ? 0 : j + 1; 

        int sum = 
            shared_memory[left]+       //i-1, j-1 , was originally 0*N+left, but erased the unnecessary multiplication
            shared_memory[j]+          //i-1, j
            shared_memory[right]+      //i-1, j+1

            shared_memory[N+left]+        //i, j-1, was originally 1*N+left, erased the unnecessary multiplication here too.
            shared_memory[N+right]+       //i, j+1

            shared_memory[(N<<1)+left]+     //i+1, j-1, was originally 2*N, made it N<<1, which is the same thing.
            shared_memory[(N<<1)+j]+        //i+1, j
            shared_memory[(N<<1)+right];    //i+1, j+1

        
        //Rules of the Game.

        if(shared_memory[N + j] == 0  && sum == 3 ){
            new_table[idx]=1; //born
        }else if ( shared_memory[N + j] == 1  && (sum < 2 || sum>3 ) ){
            new_table[idx]=0; //dies - loneliness or overpopulation
        }else{
            new_table[idx] = shared_memory[N + j]; //nothing changes
        }
    }
    return;
}


int main(int argc, char **argv){

    //declarations, error checking, allocations etc.
    char *filename = argv[1];
    int N = atoi(argv[2]);
    int t = atoi(argv[3]);
    int gen = 0;
    int *table = (int *)malloc(N*N*sizeof(int));
    if (!table)
        die("Couldn't allocate memory to table");

    //CUDA
    //initialize a 1-d grid, with total of N*BLOCKLEN threads, ready to work on our array.
    dim3 threadsPerBlock(BLOCKLEN, 1);
    dim3 numBlocks(N, 1); 
    
    //CUDA timing, copy to device and allocations
    float gputime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    read_from_file(table, filename, N);
    int *d_table;
    cudaMalloc(&d_table, N*N*sizeof(int));
    int *new_table;
    cudaMalloc(&new_table, N*N*sizeof(int));
    cudaMemcpy(d_table, table, N*N*sizeof(int), cudaMemcpyHostToDevice);

    for(gen=0; gen<t; gen++){

        cudaEventRecord(start, 0);
        //juggle arrays to save space like previous implementations.

        if(gen%2==0){
            //Pass an extra argument, since we dynamically allocate memory to the shared memory array now.
            evolve<<<numBlocks, threadsPerBlock, 3*N*sizeof(int)>>>(d_table, new_table, N);
        }else{
            evolve<<<numBlocks, threadsPerBlock, 3*N*sizeof(int)>>>(new_table, d_table, N);
        }
        cudaDeviceSynchronize(); 
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gputime, start, stop);
        printf("[%d]\t %g \n",gen, gputime/1000.0f);
    }

    //get results back!

    if(t%2==1){
        cudaMemcpy(table, new_table, N*N*sizeof(int), cudaMemcpyDeviceToHost);
    }else{
        cudaMemcpy(table, d_table, N*N*sizeof(int), cudaMemcpyDeviceToHost);
    }

    //debugging print to make sure results are consistent.
    //printCells(table, N);
    write_to_file(table, filename, N);

    //NO.MEMORY.LEFT.BEHIND!
    free(table);
    cudaFree(new_table);
    cudaFree(d_table);
    return 0;
}


