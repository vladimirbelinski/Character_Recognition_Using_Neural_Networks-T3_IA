#include <bits/stdc++.h>

using namespace std;

const int SIZE = 32;
const int NEURONS = 64;
const int TEST_SIZE = 134;
const int TRAIN_SIZE = 1800;
double alpha = .1;
const double sigma = 16;

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
  
  Matrix operator+(Matrix &o){
    double r[SIZE][SIZE];
    for(int i = 0; i < SIZE; i++)
      for(int j = 0; j < SIZE; j++)
        r[i][j] = cells[i][j]+o.cells[i][j];
    return Matrix(r);
  }
  
  Matrix operator*(double scale){
    double r[SIZE][SIZE];
    for(int i = 0; i < SIZE; i++)
      for(int j = 0; j < SIZE; j++)
        r[i][j] = cells[i][j]*scale;
    return Matrix(r);
  }
  
  void print(){
    for(int i = 0; i < SIZE; i++){
      for(int j = 0; j < SIZE; j++)
        printf("%d ",(int)(cells[i][j]+0.5));
      printf("\n");
    }
  }
};

double sq_euclidean_distance(Matrix &a,Matrix &b){
  double sq_distance = 0.;
  for(int i = 0; i < SIZE; i++){
    for(int j = 0; j < SIZE; j++){
      double dij = a.cells[i][j] - b.cells[i][j];
      sq_distance += dij*dij;
    }
  }
  return sq_distance;
}

vector<Matrix> train,test,neurons;
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

void init_neurons(){
  const int interval = 2;  
  srand(time(NULL));
  for(int i = 0; i < NEURONS; i++){
    double matrix[SIZE][SIZE];    
    for(int j = 0; j < SIZE; j++)
      for(int k = 0; k < SIZE; k++)
        matrix[j][k] = rand() % interval;    
    neurons.push_back(Matrix(matrix));
  }
}

void train_neurons(){
  for(int i = 0; i < TRAIN_SIZE; i++){        
    //searching for the closest neuron.    
    Matrix BMU = neurons[0];
    double min_dist = sq_euclidean_distance(train[i],BMU);
    for(int j = 1; j < NEURONS; j++){      
      double dist_ji = sq_euclidean_distance(train[i],neurons[j]);
      if(dist_ji < min_dist){        
        BMU = neurons[j];
        min_dist = dist_ji;
      }
    }  
    //updating the neighboring of BMU.         
    for(int j = 0; j < NEURONS; j++){       
      double dist = sq_euclidean_distance(BMU,neurons[j]);                     
      Matrix shift = (train[i]-neurons[j])*pow(M_E, -dist/sigma)*alpha;
      neurons[j] = neurons[j] + shift;          
    }
  }
}

//argv[1] = training set
//argv[2] = testing set
int main(int argc, char const *argv[]) {  
  //reading training set
  FILE *training_file = fopen(argv[1],"r");
  read_digits(train,TRAIN_SIZE,training_file);
  fclose(training_file);
  
  //reading testing set
  FILE *testing_file = fopen(argv[2],"r");
  read_digits(test,TEST_SIZE,testing_file);  
  fclose(testing_file);  
  
  init_neurons();     
  for(int l = 0; alpha > 0; l++ ){
    train_neurons();  
    for(int i = 0; i < NEURONS; i++){
      int mn = 0;
      double mn_dist = sq_euclidean_distance(neurons[i],train[mn]);
      for(int j = 1; j < TRAIN_SIZE; j++){
        double dist = sq_euclidean_distance(neurons[i],train[j]);
        if(mn_dist > dist){
          mn = j;
          mn_dist = dist;
        }
      }
      printf("%d\n",i);
      neurons[i].print();
      printf("\n");
      train[mn].print();
      printf("\n");      
    }
    alpha -= .001;
  }
  return 0;
}