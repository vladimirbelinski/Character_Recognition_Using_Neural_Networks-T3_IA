#include <bits/stdc++.h>

using namespace std;

const int SIZE = 20;

int mat1[SIZE][SIZE];
int main(void){
  for(int i = 0; i < SIZE; i++)
    for(int j = 0; j < SIZE; j++)
      scanf("%d",&mat1[i][j]);
  double correct = 0;
  for(int i = 0; i < SIZE; i++)
    for(int j = 0,k; j < SIZE; j++){
        scanf("%d",&k);
        correct += (k == mat1[i][j]);
    }
  printf("%lf%%\n",(correct*100)/(SIZE*SIZE));
  return 0;
}