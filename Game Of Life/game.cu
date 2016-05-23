/*

Conway's Game of Life CUDA implementation in C, using only global memory, with enough threads to process every cell in the grid.
Papavasileiou Themis 24/01/2015 

*/


#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/time.h>
#include <unistd.h>



struct timeval startwtime, endwtime;
double seq_time;

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
        printf("Error: %s\n", message);
    }
    return;
}

void write_to_file( int *X, char *filename, int N)
{
  FILE *fp;
  char newfilename[100];
  sprintf(newfilename,"GameOfLifecuda%dX%d.bin",N,N);
  if ( !(fp=fopen(newfilename,"w+")))
  {
    die(" couldn't open file");
  }
  if( ! fwrite(X,sizeof(int),N*N,fp))
  {
    die("couldn't really write");
  }

  fclose(fp);
}

void read_from_file( int *X, char *filename, int N)
{
    FILE *fp = fopen(filename,"r+");

    int size = fread(X,sizeof(int),N*N,fp);

    //printf("elements: %d\n",size);

    fclose(fp);

}
/*
  
  The logic is the same as in the serial version, but modified to fit the CUDA implementation needs.

*/
__global__
void evolve(int *table, int N,int *new_table)
{
    //Global memory indices.
    int i = (blockIdx.x*blockDim.x)+threadIdx.x;
    int j = (blockIdx.y*blockDim.y)+threadIdx.y;

    int k=0,sum=0;

    // if the above indices are in bounds
     if(i<N && j<N)
    {
        
            sum=0;

           if( (i!=0 && j!=0) && (i!=N-1 && j!=N-1) ) 
           {
              
               for( k=0 ; k<3 ; ++k)
               {
                  sum+= table[ (i-1)*N + (j-1+k) ]+
                        table[ (i+1)*N + (j-1+k)]+
                        table[ (i*N) + (j-1+k) ];
               }

                  sum-= table[i*N+j];
           }

           else
           {
               //printf("edge neighbors of %d , %d are ->\n",i,j);
               for(k=0 ; k<3; ++k)
               {
                   
                    sum+= table[ (i*N) + (j-1+k<0 ? N-1 : j-1+k>N-1 ? 0 : j-1+k ) ]+
                          table[ (i-1<0?N-1:i-1)*N + ( j-1+k<0 ? N-1 : j-1+k>N-1 ? 0 : j-1+k ) ]+
                          table[ (i+1>N-1?0:i+1)*N + ( j-1+k<0 ? N-1 : j-1+k>N-1 ? 0 : j-1+k ) ];
               }
                    sum-= table[i*N+j]; //subtract self-value because it's counted as a neighbor.
           }

           if( table[i*N+j] == 0 && sum == 3 ) 
              {
                new_table[i*N+j]=1;
              }
              else if( table[i*N+j] == 1 && ( sum<2 || sum>3 ) )
              {
                new_table[i*N+j]=0;
              }
              else 
              {
               new_table[i*N+j]=table[i*N+j];
              }
    }

    return;
}

int main (int argc, char **argv)
{
    //Necessary variables and float to get the time it took the GPU to compute the evolve() function.
    float gputime;  
    char *filename=argv[1];
    int N = atoi(argv[2]);
    int generations=atoi(argv[3]);

    //Find the nearest power of 2, to our grid size N.
    int nearestPower =0;

    while(true)
    {
      if( 1<<nearestPower < N )
      {
        nearestPower++;
        continue;
      }
      break;
    }

    int t = atoi(argv[4]);
    //user defines threads/block number
    dim3 threadsPerBlock(t,t); 
    //Make sure we have enough threads to cover all our cells.
    dim3 numBlocks((1<<nearestPower)/threadsPerBlock.x,(1<<nearestPower)/threadsPerBlock.y);
    
    int *table=(int *)malloc(N*N*sizeof(int));
    
    read_from_file(table, filename , N);

    // Tables for our GPU.
    int *new_table;
    cudaMalloc(&new_table,N*N*sizeof(int));

    int *device_table;
    cudaMalloc(&device_table,N*N*sizeof(int));
    
    int i;
    cudaMemcpy(device_table,table,N*N*sizeof(int),cudaMemcpyHostToDevice);  
    //print our arguments in whatever way fits our parser for the preprocessing of report data :).
    printf("%d %d \n",N,t);
    printf("\n");

    for(i=0;i<generations;++i)
    {
      /*
        
        Some implementations use a third temp array in order to feed it to the next iteration of the Game Of Life function.
        Here we save space by juggling the new_table, and device_table interchangeably, as arguments to the function.

      */

      //make sure the GPU execution is timed. ( Pulled by NVIDIA forums.)
      cudaEvent_t start,stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      cudaEventRecord(start,0);
      //The actual juggling. 
      if(i%2==0)
      {
      evolve<<<numBlocks,threadsPerBlock>>>(device_table,N,new_table);
      }
      else
      {
       evolve<<<numBlocks,threadsPerBlock>>>(new_table,N,device_table);
      }
      cudaDeviceSynchronize();
      cudaEventRecord(stop,0);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&gputime,start,stop);
      cudaEventDestroy(start);
      cudaEventDestroy(stop) ;   

      printf("%g \n",gputime/1000.0f);  
      
    }

    /*
        
        Now, depending if the number of generations was even or odd, we must get the correct array of values.

    */
    if(generations%2==1)
    {
      cudaMemcpy(table,new_table,N*N*sizeof(int),cudaMemcpyDeviceToHost);
    }
    else
    {
      cudaMemcpy(table,device_table,N*N*sizeof(int),cudaMemcpyDeviceToHost);
    }

    //Write it back to file!
    write_to_file(table,filename,N);

    //No memory left behind, on any device!
    cudaFree(new_table);
    cudaFree(device_table);
    free(table);
}


