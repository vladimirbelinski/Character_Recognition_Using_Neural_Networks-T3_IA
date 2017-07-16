# Character_Recognition_Using_Neural_Networks-T3_IA
Character Recognition Using Neural Networks

 **Autores**: Gabriel Batista Galli, Matheus Antonio Venancio Dall'Rosa e Vladimir Belinski
 
 **Descrição**: O presente arquivo faz parte da resolução do Trabalho III do CCR Inteligência Artificial, 2017-1, do curso de Ciência da Computação da Universidade Federal da Fronteira Sul - UFFS, o qual consiste na implementação de um algoritmo para reconhecimento de digitos utilizando mapas auto-organizaveis (rede de Kohonen).

## Compilação e execução

- `make train_save SAVE_TO=filename` realiza o treinamento de uma rede de Kohonen utilizando como entrada os arquivos padrões (`optdigits-orig.{tes,tra}`) e salva a rede em `filename`;

- `make load_test LOAD_FROM=filename` realiza o carregamento da rede de Kohonen salva em `filename` e a testa;

- `make train_test` realiza o treinamento de uma rede de Kohonen utilizando como entrada os arquivos padrões (`optdigits-orig.{tes,tra}`) e a testa;

- `train_test_save SAVE_TO=filename` realiza o treinamento de uma rede de Kohonen utilizando como entrada os arquivos padrões (`optdigits-orig.{tes,tra}`), a testa e salva o resultado em `filename`.

Cabe destacar que os parâmetros `SAVE_TO` e `LOAD_FROM` são opcionais. Caso não sejam fornecidos, assumem os valores padrão `network.tra` e `best_yet.tra`, respectivamente.