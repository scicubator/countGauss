/* This function has been automatically generated by whtgen. */

/*
  apply_small2( <wht>, <S>, <x> )
  ------------------------------
  computes
    WHT_(2^2) with stride <S>
  on the vector <x>
*/

#include "spiral_wht.h"

void apply_small2(Wht *W, long S, wht_value *x)
{
  wht_value t0;
  wht_value t1;
  wht_value t2;
  wht_value t3;




  t0 = x[0] + x[S];
  t1 = x[0] - x[S];
  t2 = x[2 * S] + x[3 * S];
  t3 = x[2 * S] - x[3 * S];
  x[0] = t0 + t2;
  x[2 * S] = t0 - t2;
  x[S] = t1 + t3;
  x[3 * S] = t1 - t3;
}