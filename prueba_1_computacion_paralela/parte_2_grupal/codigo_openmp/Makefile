CC = g++
CFLAGS = -O3 -fopenmp
TARGET = normalizacion
SRC = normalizacion.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) -lm

clean:
	rm -f $(TARGET)
