#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "spiral_wht.h"


/* 
 * Computes fast hadamard transform 
 */
void anchor_fast( double *W, int m, int n);

void anchor_fast( double *W, int m, int n)
{
  Wht *Wht_tree;
  int i,j,k;
  clock_t cp0= clock();
  clock_t cp1= clock();
  wht_value x[n]; /* a vector of doubles */
  /* initialize somehow */
   //x[n] = (n & 1) * 2 - 1; 
    /* get wht tree for 2^14 */
        Wht_tree = wht_get_tree((int) log2(n));

    //#ifdef PARA_ON
    ///* if PARA is enabled, get a parallel wht tree
    //     with default thread number */
    //Wht_tree = wht_get_p_tree(2, THREADNUM);
    //#endif

    for(j=0 ; j< m ; j++)
    {    
        for(i = 0; i < n; i++) 
            x[i] = W[j*n+i]; 

        /* compute wht_{2^14} */
        wht_apply(Wht_tree, 1, x);  /* stride is 1 */
        for(k = 0; k < n; k++) 
            W[j*n+k]=x[k]; 
    }  
    wht_delete(Wht_tree);
cp1= clock();
printf("CPU Time %.2f\n", ((float)(cp1-cp0))/CLOCKS_PER_SEC);
  return ;
}


int main(int argc, char *argv[])
{
int m,n;
long i;
double *W;
if(argc!=3)
    return 0;
m=atoi(argv[1]);
n=atoi(argv[2]);

W=malloc(sizeof(double)*m*n);
if(W==NULL)
    return 0;

for(i=0; i < m*n ; ++i)
    W[i] = (m*n & 1) * 2 - 1; 


/* Get a solver object and initialize the dose matrix, prescription */
clock_t cp0= clock();

clock_t cp1= clock();
anchor_fast(W,m,n);
cp1= clock();
printf("CPU Time %.2f\n", ((float)(cp1-cp0))/CLOCKS_PER_SEC);

return 0;
}
 
