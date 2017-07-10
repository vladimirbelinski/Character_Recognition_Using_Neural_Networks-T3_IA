STD_C = -std=c++11
FLAGS = -Wall -O2

all: kohonen

train0: all
	./kohonen --tra optdigits-orig.tra --snet network.tra

train: all
	./kohonen --lnet network.tra --tra optdigits-orig.tra --snet network.tra

load_test: all
	./kohonen --lnet network.tra --tes optdigits-orig.tes
	
train_test: all
	./kohonen --tra optdigits-orig.tra --tes optdigits-orig.tes

train_test_save: all
	./kohonen --tra optdigits-orig.tra --tes optdigits-orig.tes --snet network.tra
	
kohonen: main.cpp
	g++ main.cpp -o kohonen ${STD_C} ${FLAGS}

clean:
	rm kohonen
