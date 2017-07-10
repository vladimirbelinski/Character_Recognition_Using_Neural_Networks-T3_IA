#include <bits/stdc++.h>

using namespace std;

const int SIZE = 32;
const int NEURONS = 20;
const int TRAIN_ITER = 400;
const double sigma = 0.8;

double alpha = .05;

#define f first
#define s second

typedef pair<int,int> ii;

struct Matrix{
  int digit;
  double cells[SIZE][SIZE];
  Matrix(){}
  Matrix(double _cells[SIZE][SIZE]){
    memcpy(cells,_cells,sizeof cells);
  }
  Matrix(double _cells[SIZE][SIZE],int _digit){
    digit = _digit;
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
void read_digits(vector<Matrix> &m,FILE *src){
  int a;
  fscanf(src,"%d\n",&a);  
  for (int i = 0,c; i < a; i++) {
    double digit[SIZE][SIZE];
    for(int j = 0; j < SIZE; j++){
      for(int k = 0; k < SIZE; k++){
        fscanf(src," %c",&c);
        digit[j][k] = c-'0';
      }
    }   
    fscanf(src,"  %c ",&c);
    m.push_back(Matrix(digit,c-'0'));
  }
}

void init_neurons(){
  printf("Inicializando rede.\n");
  srand(time(NULL));
  for(int i = 0; i < NEURONS; i++)
    for(int i2 = 0; i2 < NEURONS; i2++)
      for(int j = 0; j < SIZE; j++)
        for(int k = 0; k < SIZE; k++)
          neurons[i][i2][j][k] = ((double)rand()) / RAND_MAX;
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
  vector<int> training_order = randomic_order(train.size());  
  for(int i = 0; i < (int)train.size(); i++){        
    ii BMU = closest_neuron(train[training_order[i]]);
    //updating the neighborhood of BMU.
    for(int j = 0; j < NEURONS; j++){
      double dist = sq_euclidean_distance(BMU,ii(j,BMU.s));        
      if(dist > 20.*sigma ) continue;
      for(int j2 = 0; j2 < NEURONS; j2++){
        double dist = sq_euclidean_distance(BMU,ii(j,j2));        
        if(dist > 20.*sigma ) continue;        
        Matrix shift = (train[training_order[i]]-neurons[j][j2])*pow(M_E, -dist/sigma)*alpha;
        neurons[j][j2] = neurons[j][j2] + shift;
      }
    }
  }
}

int matches[NEURONS][NEURONS];
void print_matches(){
  int frequence[10];
  memset(frequence,0,sizeof(frequence));
  for(int i = 0; i < NEURONS; i++){
    for(int i2 = 0; i2 < NEURONS; i2++){
      int mn = 0;
      double mn_dist = sq_euclidean_distance(neurons[i][i2],train[mn]);
      for(int j = 1; j < (int)train.size(); j++){
        double dist = sq_euclidean_distance(neurons[i][i2],train[j]);
        if(mn_dist > dist){
          mn = j;
          mn_dist = dist;
        }
      }
      //printf("%d %d\n",i,i2);
      //neurons[i][i2].print(true);
      //printf("\n");
      //train[mn].print(true);
      frequence[train[mn].digit]++;
      matches[i][i2] = train[mn].digit;
      //printf("\n");
    }
  }
  for(int i = 0; i < 10; i++) printf("%d %d\n",i,frequence[i]);
  for(int i = 0; i < NEURONS; i++){
    for(int i2 = 0; i2 < NEURONS; i2++)
      printf("%d ",matches[i][i2]);
    printf("\n");
  }
}

void load_neurons(string dir){
  printf("Carregando rede.\n");
  FILE *neurons_file = fopen(dir.c_str(),"r");
  for(int i = 0; i < NEURONS; i++)
    for(int i2 = 0; i2 < NEURONS; i2++) {
      double w;
      double digit[SIZE][SIZE];
      for(int j = 0; j < SIZE; j++)
        for(int k = 0; k < SIZE; k++){
          fscanf(neurons_file," %lf",&w);
          digit[j][k] = w;
        }
      neurons[i][i2] = Matrix(digit);
    }
  fclose(neurons_file);
}

void save_neurons(string dir){
  printf("Salvando rede.\n");
  FILE *neurons_file = fopen(dir.c_str(),"w");
  for(int i = 0; i < NEURONS; i++)
    for(int i2 = 0; i2 < NEURONS; i2++){
      for(int j = 0; j < SIZE; j++){
        for(int k = 0; k < SIZE; k++)
          fprintf(neurons_file," %lf",neurons[i][i2][j][k]);
        fprintf(neurons_file,"\n");
      }
      fprintf(neurons_file,"\n");
    }
  fclose(neurons_file);
}

//--tes path/to/test/file
//--tra path/to/training/file
//--lnet path/to/load/trained_neurons/file
//--snet path/to/save/trained_neurons/file
int main(int argc, char const *argv[]) {
  map<string,string> param;
  for(int i = 1; i < argc; i += 2) param[argv[i]] = argv[i+1];
  if(param.find("--lnet") != param.end()) load_neurons(param["--lnet"]);
  else init_neurons();
  if(param.find("--tra") != param.end()){
    //reading training set
    FILE *training_file = fopen(param["--tra"].c_str(),"r");
    read_digits(train,training_file);
    fclose(training_file);
    for(int l = 0; alpha > 0 && l < TRAIN_ITER; l++, alpha -= .00001)
      train_neurons();
    print_matches();
  }
  if(param.find("--tes") != param.end()){
    //reading testing set
    FILE *testing_file = fopen(param["--tes"].c_str(),"r");
    read_digits(test,testing_file);
    fclose(testing_file);
  }
  if(param.find("--snet") != param.end()) save_neurons(param["--snet"]);
  return 0;
}
