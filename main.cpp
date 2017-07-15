/* Arquivo: main.cpp
   Autores: Gabriel Batista Galli, Matheus Antonio Venancio Dall'Rosa e Vladimir Belinski
   Descrição: O presente arquivo faz parte da resolução do Trabalho III do CCR Inteligência Artificial, 2017-1, do curso de
              Ciência da Computação da Universidade Federal da Fronteira Sul - UFFS, o qual consiste na implementação de
              um algoritmo para reconhecimento de digitos utilizando mapas auto-organizaveis (rede de Kohonen).
              --> main.cpp é o arquivo principal do trabalho, onde é encontrada a implementação do algoritmo para reconhecimento.
*/
#include <bits/stdc++.h>

using namespace std;

const int SIZE = 32;
const int NEURONS = 32;
const int TRAIN_ITER = 1000;
double sigma = 2.666;

double alpha = .1;

#define f first
#define s second

typedef pair<int,int> ii;

struct Matrix{
  int digit;
  int digits[10];
  double cells[SIZE][SIZE];
  Matrix(){
    digit = -1;
    memset(digits,0,sizeof(digits));
  }
  Matrix(double _cells[SIZE][SIZE]){
    digit = -1;
    memset(digits,0,sizeof(digits));
    memcpy(cells,_cells,sizeof cells);
  }
  Matrix(double _cells[SIZE][SIZE],int _digit){
    digit = _digit;
    memset(digits,0,sizeof(digits));
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

vector<Matrix> train;
Matrix neurons[NEURONS][NEURONS];
void print_neuron_digits(){
  for(int i = 0; i < NEURONS; i++){
    for(int j = 0; j < NEURONS; j++){
      if(j) printf(" ");
      if(neurons[i][j].digit == -1) printf("_");
      else printf("%d",neurons[i][j].digit);
    }
    printf("\n");
  }
}

void read_digits(vector<Matrix> &m,FILE *src){
  int a;
  char line[SIZE+10];
  fscanf(src,"%d\n",&a);
  for (int i = 0; i < a; i++) {
    double digit[SIZE][SIZE];
    for(int j = 0; j < SIZE; j++){
      fgets(line,SIZE+10,src);
      for(int k = 0; k < SIZE; k++)
        digit[j][k] = line[k]-'0';
    }
    fgets(line,SIZE,src);
    m.push_back(Matrix(digit,line[1]-'0'));
  }
}

void init_neurons(){
  printf("Initializing network.\n");
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

void match_training(vector<Matrix> & t){
  for(int k = 0; k < (int)t.size(); k++){
    ii BMU;
    double BMU_dist = 401.;
    for(int i = 0; i < NEURONS; i++){
      for(int i2 = 0; i2 < NEURONS; i2++){
        double dist = sq_euclidean_distance(neurons[i][i2],t[k]);
        if(dist < BMU_dist){
          BMU = ii(i,i2);
          BMU_dist = dist;
        }
      }
    }
    neurons[BMU.f][BMU.s].digits[t[k].digit]++;
  }
  for(int i = 0; i < NEURONS; i++)
    for(int i2 = 0; i2 < NEURONS; i2++){
      int mx_d = -1;
      int mx_ocurr = 0;
      for(int k = 0; k < 10; k++)
        if(neurons[i][i2].digits[k] > mx_ocurr){
          mx_d = k;
          mx_ocurr = neurons[i][i2].digits[k];
        }
      neurons[i][i2].digit = mx_d;
    }
}

void run_test(vector<Matrix> &t){
  int correct_occurrences = 0;
  int digit_correct_occurrences[10], digit_occurrences[10];
  memset(digit_occurrences,0,sizeof(digit_occurrences));
  memset(digit_correct_occurrences,0,sizeof(digit_correct_occurrences));
  for(int k = 0; k < (int)t.size(); k++){
    ii BMU;
    double BMU_dist = 401.;
    for(int i = 0; i < NEURONS; i++){
      for(int i2 = 0; i2 < NEURONS; i2++){
        double dist = sq_euclidean_distance(neurons[i][i2],t[k]);
        if(dist < BMU_dist){
          BMU = ii(i,i2);
          BMU_dist = dist;
        }
      }
    }
    int matched = (neurons[BMU.f][BMU.s].digit == t[k].digit);
    correct_occurrences += matched;
    digit_occurrences[t[k].digit]++;
    digit_correct_occurrences[t[k].digit] += matched;
  }
  printf("Number of matches: %lf%%\n",((double)correct_occurrences*100)/(double)t.size());
  for(int i = 0; i < 10; i++)
    printf("Digit %d ocurred %d times with an acurracy of %lf%%\n",i,digit_occurrences[i],(double)digit_correct_occurrences[i]*100./digit_occurrences[i]);
}

void load_neurons(string dir){
  printf("Loading network.\n");
  FILE *neurons_file = fopen(dir.c_str(),"r");
  for(int i = 0; i < NEURONS; i++)
    for(int i2 = 0; i2 < NEURONS; i2++){
      for(int j = 0; j < SIZE; j++)
        for(int k = 0; k < SIZE; k++)
          fscanf(neurons_file," %lf",&neurons[i][i2][j][k]);
      fscanf(neurons_file," %d",&neurons[i][i2].digit);
    }
  fclose(neurons_file);
}

void save_neurons(string dir){
  printf("Saving network.\n");
  FILE *neurons_file = fopen(dir.c_str(),"w");
  for(int i = 0; i < NEURONS; i++)
    for(int i2 = 0; i2 < NEURONS; i2++){
      for(int j = 0; j < SIZE; j++){
        fprintf(neurons_file,"%lf",neurons[i][i2][j][0]);
        for(int k = 1; k < SIZE; k++)
          fprintf(neurons_file," %lf",neurons[i][i2][j][k]);
        fprintf(neurons_file,"\n");
      }
      fprintf(neurons_file,"%d\n\n",neurons[i][i2].digit);
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
  if(param.find("--lnet") != param.end()){
    load_neurons(param["--lnet"]);
    print_neuron_digits();
  }
  else{
    init_neurons();
    if(param.find("--tra") != param.end()){
      //reading training set
      FILE *training_file = fopen(param["--tra"].c_str(),"r");
      read_digits(train,training_file);
      fclose(training_file);
      for(int l = 0; alpha > 0 && l < TRAIN_ITER; l++, alpha -= .00009, sigma -= .0021){
        printf("Iteration: %d, alpha: %lf, sigma: %lf\n",l,alpha,sigma);
        train_neurons();
        if((l+1)%(TRAIN_ITER/5) == 0){
          match_training(train);
          print_neuron_digits();
        }
      }
      match_training(train);
      print_neuron_digits();
    }
  }
  if(param.find("--tes") != param.end()){
    vector<Matrix> test;
    //reading testing set
    FILE *testing_file = fopen(param["--tes"].c_str(),"r");
    read_digits(test,testing_file);
    fclose(testing_file);
    run_test(test);
  }
  if(param.find("--snet") != param.end()) save_neurons(param["--snet"]);
  return 0;
}
