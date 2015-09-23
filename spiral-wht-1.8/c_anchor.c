#include <math.h>
#include <stdio.h>
#include <stdlib.h>
//#include <time.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include "spiral_wht.h"
//#include <omp.h>


/* 
 * Computes fast hadamard transform 
 */
void anchor_fast( double *W, double *B, double *G, int *P, int m, int n);

void anchor_fast( double *W, double *B, double *G, int *P, int m, int n)
{
  Wht *Wht_tree;
  int i,j,k;
  //clock_t cp0= clock();
  //clock_t cp1= clock();
  //wht_value x[n]; /* a vector of doubles */
  wht_value *x= (wht_value *) malloc(sizeof(double)*n);
  if (x==NULL)
      return;
  /* initialize somehow */
   //x[n] = (n & 1) * 2 - 1; 
    /* get wht tree for 2^14 */
        //Wht_tree = wht_get_tree((int) log2(n));
        if(n>=400000) 
            Wht_tree = wht_get_tree((int) log2(1024));
        else
            Wht_tree = wht_get_tree((int) log2(n));

    //#ifdef PARA_ON
    ///* if PARA is enabled, get a parallel wht tree
    //     with default thread number */
    //Wht_tree = wht_get_p_tree(2, THREADNUM);
    //#endif

    //# pragma omp parallel \
    //shared (W, B, G, P) \
    //private (i,j,k)
   
    //# pragma omp for 
    for(j=0 ; j< m ; j++)
    {    
        for(i = 0; i < n; i++) 
            x[i] = B[i]*W[j*n+i]; 

        /* compute wht_{2^14} */
        wht_apply(Wht_tree, 1, x);  /* stride is 1 */
        for(k = 0; k < n; k++) 
            W[j*n+k]=G[k]*x[P[k]]; 
        wht_apply(Wht_tree, 1, (wht_value *) &W[j*n]);  /* stride is 1 */
    }  
    wht_delete(Wht_tree);
    free(x);
//cp1= clock();
//printf("CPU Time %.2f\n", ((float)(cp1-cp0))/CLOCKS_PER_SEC);
  return ;
}


int main(int argc, char *argv[])
{
int m,n;
int i;
double *W, *B, *G, val;
int *P;
double sigma=1.0; // variance 1

const gsl_rng_type *T;
gsl_rng *r;

if(argc!=3)
    return 0;
m=atoi(argv[1]);
n=atoi(argv[2]);

gsl_rng_env_setup();
T = gsl_rng_default;
r = gsl_rng_alloc (T);

W=malloc(sizeof(double)*m*n);
B=malloc(sizeof(double)*n);
G=malloc(sizeof(double)*n);
P=malloc(sizeof(int)*n);
if(W==NULL || B==NULL || G==NULL)
    return 0;

for(i=0; i < m*n ; ++i)
    W[i] = (m*n & 1) * 2 - 1; 

for(i=0; i < n  ; ++i)
{    
    val = rand()/((double)RAND_MAX + 1);  
    if (val>0.5)
        B[i] =  1.0;
    else
        B[i] = -1.0;
}
for(i=0; i < n  ; ++i)
    G[i] = gsl_ran_gaussian(r,sigma);  

for(i=0; i < n  ; ++i)
    P[i] = (int) i;

/* Get a solver object and initialize the dose matrix, prescription */
//clock_t cp0= clock();

//clock_t cp1= clock();
anchor_fast(W,B,G,P,m,n);
//cp1= clock();
//printf("CPU Time %.2f\n", ((float)(cp1-cp0))/CLOCKS_PER_SEC);

free(W);
free(B);
free(G);
//free(P);

return 0;
}
 
