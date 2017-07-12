STD_C = -std=c++11
FLAGS = -Wall -O2
SAVE_TO = network.tra
LOAD_FROM = network.tra

all: kohonen

train0: all
	./kohonen --tra optdigits-orig.tra --snet ${SAVE_TO}

train: all
	./kohonen --lnet ${LOAD_FROM} --tra optdigits-orig.tra --snet ${SAVE_TO}

load_test: all
	./kohonen --lnet ${LOAD_FROM} --tes optdigits-orig.tes
	
train_test: all
	./kohonen --tra optdigits-orig.tra --tes optdigits-orig.tes

train_test_save: all 
	./kohonen --tra optdigits-orig.tra --tes optdigits-orig.tes --snet ${SAVE_TO}
	
kohonen: main.cpp
	g++ main.cpp -o kohonen ${STD_C} ${FLAGS}

clean:
	rm kohonen
