#include <bits/stdc++.h>

using namespace std;

const int SIZE = 32;
const int TEST_SIZE = 134;
const int TRAIN_SIZE = 1800;

struct Matrix{  
  double cells[SIZE][SIZE];
  Matrix();
  Matrix(double _cells[SIZE][SIZE]){
    memcpy(cells,_cells,sizeof cells);
  }
  Matrix operator-(Matrix &o){
    double r[SIZE][SIZE];
    for(int i = 0; i < SIZE; i++)
      for(int j = 0; j < SIZE; j++)
        r[i][j] = cells[i][j]-o.cells[i][j];
    return Matrix(r);
  }
  void square_elements(){
    for(int i = 0; i < SIZE; i++)
      for(int j = 0; j < SIZE; j++)
        cells[i][j] *= cells[i][j];
  }
};

vector<Matrix> train,test;
void read_digits(vector<Matrix> &m,int a,FILE *src){
  for (int i = 0,c; i < a; i++) {    
    double digit[SIZE][SIZE];
    for(int j = 0; j < SIZE; j++){
      for(int k = 0; k < SIZE; k++){        
        fscanf(src," %c",&c);
        digit[j][k] = c-'0';        
      }      
    }
    fscanf(src,"  %c ",&c);
    m.push_back(digit);
  }
}

int main(int argc, char const *argv[]) {
  //argv[1] = training set
  //argv[2] = testing set
  
  //reading training set
  FILE *training_file = fopen(argv[1],"r");
  read_digits(train,TRAIN_SIZE,training_file);
  fclose(training_file);
  
  //reading testing set
  FILE *testing_file = fopen(argv[2],"r");
  read_digits(test,TEST_SIZE,testing_file);  
  fclose(testing_file);
  
  return 0;
}