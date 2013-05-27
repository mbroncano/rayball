SRC=main.cpp
CC=g++
CCFLAGS=-framework OpenGl -framework Glut -fopenmp -g


rayball: $(SRC) Makefile
	$(CC) $(CCFLAGS) $(SRC) -o rayball

all: rayball

clean:
	rm -rf rayball
