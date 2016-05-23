/*

Conway's Game of Life serial implementation in C
Papavasileiou Themis 24/01/2015 

*/

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/time.h>
#include <unistd.h>


// variables to time the execution.
struct timeval startwtime, endwtime;
double seq_time;

/*
  die and warn functions taken from Zed E. Shaw's Learn C the hard way.
*/

/**
 * die - display an error and terminate.
 * Used when some fatal error happens
 * and continuing would mess things up.
 */
void die(const char *message){
    if(errno){
        perror(message);
    }else{
        printf("Error: %s\n", message);
    }
    exit(1);
}

/**
 * warn - display a warning and continue
 * used when something didn't go as expected
 */
void warn(const char *message){
    if(errno){
        perror(message);
    }else{
        printf("Error: %s\n", message);
    }
    return;
}

/*
  Write to file function, with error handling.
*/
void write_to_file( int *X, char *filename, int N)
{
  FILE *fp;
  char newfilename[100];
  sprintf(newfilename,"GameOfLifeResults%dX%d.bin",N,N);
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
/*
  Read from file function. No debugs here.
*/

void read_from_file( int *X, char *filename, int N)
{
    FILE *fp = fopen(filename,"r+");

    int size = fread(X,sizeof(int),N*N,fp);

    printf("elements: %d\n",size);

    fclose(fp);

}

/*
  Function that actually implements the Game Of Life.
  Sacrificed some readability for performance. 
*/
void evolve(int *table, int N)
{
    //debug message
    /*printf("in evolve\n"); */

    int i,j,k,d=0,b=0,o=0,sum=0;
    int * new = (int *)malloc(N*N*sizeof(int));
    
    /*
      Scary Loop ahead!
      The logic is that for a given grid, we're either on the edge rows/columns (ie row=0/column=0 etc etc) or on the 'inside'.
      We loop over the grid, and for each cell we sum its neighbors, since it's only 0s and 1s, so we know how many neighbors it has alive.
      Then according to the rules we decide the state of said cell for the next generation. 

      One of the requirements of the game is that the changes should be made all at once, so we can't update cell values on the fly.
      For that a second array is needed, so that we copy the entire generation's next state in it, and return it, the change seems instant.

      Note that cyclic boundaries are required too, so edge rows/columns need special treatment to find their neighboring cells.
    */
    for (i=0 ; i<N ; ++i)
    {
        for(j=0 ; j<N ; ++j)
        {
            //make sure each cell gets correctly initialized sum.
            sum=0;
           // If we're not on top row/column AND not on bottom row/column then we must be on the 'inside' of the array.
           if( (i!=0 && j!=0) && (i!=N-1 && j!=N-1) ) 
           {
              // printf(" neighbors of %d , %d are \n",i,j); //debug to check if neighbors are computed correctly on small grids.

              /*
              The sum is computed as such:

              For every cell we need 8 neighbors.3 in the above row, 2 in the same row and 3 in the bottom row.
              We start from the upper left neighbor. Its coordinates are (i-1),(j-1). To get all the neighbors in the above 
              we have to consider also (i-1),(j)(the cell exactly above) , (i-1),(j+1) ( the cell above and to the right).

              That's achieved by the loop below, by taking j-1 and adding 0,1,2 to it, making sure that whatever row we're in, the 3 columns we want 
              are considered.

              This process is done for our own row (i),for the row above us (i-1) for the row below us(i+1). Now when we consider our own row, 
              the value of our own cell is counted in the sum, and that's not supposed to happen. To correct for this error, when the looping is done,
              we subtract our self-value from the sum.

              */

               for( k=0 ; k<3 ; ++k)
               {
                  sum+= table[ (i-1)*N + (j-1+k) ]+
                        table[ (i+1)*N + (j-1+k)]+
                        table[ (i*N) + (j-1+k) ];
               }
               sum-=table[i*N+j];
           }
           else
           {
               /*
                  The same logic applies here, but now since we're not on the inside, we must be on the 'edge' rows/columns of the array. 

                  So extra care for the cyclic boundaries must be taken. To make the code compact the if then else statement is used.
                  Also note that we use the array[i*N+j] notation. To better understand the sum below let's analyze one line.
                  table[ ( i-1<0? N-1 : i-1 )*N + ( j-1+k<0 ? N-1 : j-1+k>N-1 ? 0 : j-1+k ) ]
                  This is equivalent to table[ i*N + j], with i=( i-1<0? N-1 : i-1 )
                                                              j=( j-1+k<0 ? N-1 : j-1+k>N-1 ? 0 : j-1+k ) 
                  And for example what [j-1+k<0 ? N-1 : j-1+k>N-1 ? 0 : j-1+k ] says is:
                    Since you're on the edge column, check if j-1+k<0, ( meaning can I move in the column and still be in the array? ).
                      >If I can't, then this must mean I'm at the leftmost column and I must look for neighbors in the rightmost (N-1) one.

                      >If I can, then move j-1+k but again check if I that move takes me out of the N-1 boundary. 

                        >If it does, then it means I'm in the rightmost column and I must look for neighbors in the leftmost one (0).

                        >If it doesn't, then it's business as usual, so I move j-1+k.
               */
               for(k=0 ; k<3; ++k)
               {
                   
                  sum+= table[ ( i*N ) + ( j-1+k<0 ? N-1 : j-1+k>N-1 ? 0 : j-1+k ) ]+
                        table[ ( i-1<0? N-1 : i-1 )*N + ( j-1+k<0 ? N-1 : j-1+k>N-1 ? 0 : j-1+k ) ]+
                        table[ ( i+1>N-1 ? 0 : i+1 )*N + ( j-1+k<0 ? N-1 : j-1+k>N-1 ? 0 : j-1+k ) ];
               }
               sum-=table[i*N+j]; //subtract self-value because it's counted as a neighbor.
           }

           /*
            Rules of the game below. Check current value and sum of neighbors, then decide accordingly. 

            There are some counters below, to ensure that for every iteration, at least one of the Ifs is visited.
           */
           if(table[i*N+j]==0 && sum==3)
              {
                new[i*N+j]=1;
                b++;
              }
              else if(table[i*N+j]==1 && (sum<2 || sum>3))
              {
                new[i*N+j]=0;
                d++;
              }
              else 
              {
               new[i*N+j]=table[i*N+j];
               o++; 
              }
           
        }

        
        

    }

    //Check and warn us any cell was left unattended, using the previously described logic.
    if( (b+d+o)!=N*N ){warn("well that's strange!\n");}

    //Copy the new table back to the old one, so change from generation to generation happens instantly.
    for(i=0;i<N;++i)
    {
      for(j=0;j<N;++j)
      {
        table[i*N+j]=new[i*N+j];
      }
    }

    // No memory left behind!
    free(new);
    return;





}

int main (int argc, char **argv)
{

    int i,j,k;
    // file to read from
    char *filename=argv[1];
    // Size of the N*N table
    int N = atoi(argv[2]);
    // Number of generations to be computed.
    int generations=atoi(argv[3]);

    printf(" reading %d x %d table from file -> %s\n",N,N,filename);

    //Table to be used to hold the cells of the game for a N*N grid. We handle it as 2d with the i*N+j notation.
    int *table=(int *)malloc(N*N*sizeof(int));
    
    read_from_file(table, filename , N);
    //wild pointer
    int *p;

    for(i=0;i<generations;++i)
    {
      gettimeofday (&startwtime, NULL);
      evolve(table,N);
      gettimeofday (&endwtime, NULL);
      //timing stops here, then we report it.
      seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
            + endwtime.tv_sec - startwtime.tv_sec);

      printf("Generation %d time -> %f\n",i, seq_time);
    }

    //write the grid after x generations.
    write_to_file(table,filename,N);

    
    // No memory left behind.
    free(table);
    return 0;
}


