/*

Conway's Game of Life CUDA implementation in C, using only global memory, with grid stride loops.

Should expect same performance with game.cu, since NVIDIA says that grid stride loops have the same
instruction cost as monolithic kernels.

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

__global__
void evolve(int *table, int N,int *new_table)
{
    int k=0,sum=0;
    /*
      
      Perform grid strides, making sure that each thread processes more than one cell.
    
    */

  for(int i = (blockIdx.x*blockDim.x)+threadIdx.x;i<N;i+=blockDim.x*gridDim.x){
    for(int j = (blockIdx.y*blockDim.y)+threadIdx.y;j<N;j += blockDim.y * gridDim.y)
    {
            
            sum=0;
           if( (i!=0 && j!=0) && (i!=N-1 && j!=N-1) ) 
           {
               for( k=0 ; k<3 ; ++k  )
               {
                  sum+= table[ (i-1)*N + (j-1+k) ]+
                        table[ (i+1)*N + (j-1+k)]+
                        table[ (i*N) + (j-1+k) ];
               }
                  sum-= table[i*N+j];
           }
           else
           {
               for( k=0 ; k<3; ++k )
               {
                   
                  sum+= table[ (i*N) + ( j-1+k<0 ? N-1 : j-1+k>N-1 ? 0 : j-1+k ) ]+
                        table[ ( i-1<0 ? N-1 : i-1 )*N + ( j-1+k<0 ? N-1 : j-1+k>N-1 ? 0 : j-1+k ) ]+
                        table[ ( i+1>N-1 ? 0 : i+1 )*N + ( j-1+k<0 ? N-1 : j-1+k>N-1 ? 0 : j-1+k ) ];
               }
                  sum-= table[i*N+j]; //subtract self-value because it's counted as a neighbor.
           }

           if( table[i*N+j] == 0 && sum == 3 )
              {
                new_table[i*N+j]=1;
              }
              else if( table[i*N+j] == 1 && ( sum<2 || sum>3 ))
              {
                new_table[i*N+j]=0;
              }
              else 
              {
               new_table[i*N+j]=table[i*N+j];
              }
    }

  }
  return;
}


int main (int argc, char **argv)
{

    float gputime;  
    char *filename=argv[1];
    int N = atoi(argv[2]);
    int generations=atoi(argv[3]);
    int t = atoi(argv[4]);
    int b = atoi(argv[5]);

    //user specifies both threads/block and total number of blocks this time.
    dim3 threadsPerBlock(t,t);
    dim3 numBlocks(b,b);

    //Necessary allocations and reads.
    printf("%d,%d\n",t,b);

    int *table=(int *)malloc(N*N*sizeof(int));
    
    read_from_file(table, filename , N);

    int *new_table;
    cudaMalloc(&new_table,N*N*sizeof(int));

    int *device_table;
    cudaMalloc(&device_table,N*N*sizeof(int));
    
    int i;
    
    cudaMemcpy(device_table,table,N*N*sizeof(int),cudaMemcpyHostToDevice);

    for(i=0;i<generations;++i)
    {
      cudaEvent_t start,stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      cudaEventRecord(start,0);

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

    if(generations%2==1)
    {
      cudaMemcpy(table,new_table,N*N*sizeof(int),cudaMemcpyDeviceToHost);
    }
    else
    {
      cudaMemcpy(table,device_table,N*N*sizeof(int),cudaMemcpyDeviceToHost);
    }

    write_to_file(table,filename,N);

    //NO MEMORY LEFT BEHIND!
    cudaFree(new_table);
    cudaFree(device_table);
    free(table);
}


