#include <bits/stdc++.h>

using namespace std;

const int SIZE = 32;
const int NEURONS = 8;
const int TEST_SIZE = 134;
const int TRAIN_SIZE = 1800;
double alpha = .1;
const double sigma = 2;

#define f first
#define s second

typedef pair<int,int> ii;

struct Matrix{ 
  double cells[SIZE][SIZE];
  Matrix(){}
  Matrix(double _cells[SIZE][SIZE]){
    memcpy(cells,_cells,sizeof cells);
  }  
  Matrix operator-(Matrix &o){
    double r[SIZE][SIZE];
    for(int i = 0; i < SIZE; i++)
      for(int j = 0; j < SIZE; j++)
        r[i][j] = cells[i][j]-o[i][j];
    return Matrix(r);
  }  
  Matrix operator+(Matrix &o){
    double r[SIZE][SIZE];
    for(int i = 0; i < SIZE; i++)
      for(int j = 0; j < SIZE; j++)
        r[i][j] = cells[i][j]+o[i][j];
    return Matrix(r);
  }  
  Matrix operator*(double scale){
    double r[SIZE][SIZE];
    for(int i = 0; i < SIZE; i++)
      for(int j = 0; j < SIZE; j++)
        r[i][j] = cells[i][j]*scale;
    return Matrix(r);
  }  
  double* operator[](int i){
    return cells[i];
  }  
  void print(bool trunc){
    for(int i = 0; i < SIZE; i++){
      for(int j = 0; j < SIZE; j++)
        if(trunc) printf("%d ",(int)(cells[i][j]+0.5));
        else printf("%.4lf ",cells[i][j]);
      printf("\n");
    }
  }
};

double sq_euclidean_distance(Matrix &a,Matrix &b){
  double sq_distance = 0.;
  for(int i = 0; i < SIZE; i++){
    for(int j = 0; j < SIZE; j++){
      double dij = a[i][j] - b[i][j];
      sq_distance += dij*dij;
    }
  }
  return sq_distance;
}

double sq_euclidean_distance(ii a,ii b){
  double dx = a.f-b.f;
  double dy = a.s-b.s;
  return dx*dx+dy*dy;
}

vector<Matrix> train,test;
Matrix neurons[NEURONS][NEURONS];
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
  srand(time(NULL));
  for(int i = 0; i < NEURONS; i++)
    for(int i2 = 0; i2 < NEURONS; i2++)
      for(int j = 0; j < SIZE; j++)
        for(int k = 0; k < SIZE; k++)
          neurons[i][i2][j][k] = (double)rand() / RAND_MAX;                         
}

ii closest_neuron(Matrix &m){
  ii BMU;
  double min_dist = SIZE*SIZE;
  for(int j = 0; j < NEURONS; j++)    
    for(int j2 = 0; j2 < NEURONS; j2++){      
      double dist_ji = sq_euclidean_distance(m,neurons[j][j2]);
      if(dist_ji < min_dist){        
        BMU = ii(j,j2);
        min_dist = dist_ji;
      }
    }
  return BMU;
}

vector<int> randomic_order(int n){
  list<int> elem;
  for(int i = 0; i < n; i++) elem.push_back(i);
  vector<int> order;
  while(!elem.empty()){
    auto it = elem.begin();    
    advance(it,rand()%elem.size());
    order.push_back(*it);
    elem.erase(it);
  }
  return order;
}

void train_neurons(){
  vector<int> training_order = randomic_order(TRAIN_SIZE);  
  for(int i = 0; i < TRAIN_SIZE; i++){                
    ii BMU = closest_neuron(train[training_order[i]]);    
    //updating the neighborhood of BMU.         
    for(int j = 0; j < NEURONS; j++){       
      for(int j2 = 0; j2 < NEURONS; j2++){        
        double dist = sq_euclidean_distance(BMU,ii(j,j2));                     
        Matrix shift = (train[i]-neurons[j][j2])*pow(M_E, -dist/sigma)*alpha;
        neurons[j][j2] = neurons[j][j2] + shift;  
      }        
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
  for(int l = 0; alpha > 0 && l < 1; l++,alpha -= .001 ){
    train_neurons();  
    for(int i = 0; i < NEURONS; i++){
      for(int i2 = 0; i2 < NEURONS; i2++){
        int mn = 0;
        double mn_dist = sq_euclidean_distance(neurons[i][i2],train[mn]);
        for(int j = 1; j < TRAIN_SIZE; j++){
          double dist = sq_euclidean_distance(neurons[i][i2],train[j]);
          if(mn_dist > dist){
            mn = j;
            mn_dist = dist;
          }
        }
        printf("%d %d\n",i,i2);
        neurons[i][i2].print(false);
        printf("\n");
        train[mn].print(true);
        printf("\n");      
      }
    }    
  }
  return 0;
}