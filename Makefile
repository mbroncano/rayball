SRC=main.cpp
CC=g++
CCFLAGS=-framework OpenGl -framework Glut -fopenmp -O3 -O2


rayball: $(SRC) Makefile
	$(CC) $(CCFLAGS) $(SRC) -o rayball

all: rayball

clean:
	rm -rf rayball
