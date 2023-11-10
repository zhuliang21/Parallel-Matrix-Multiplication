# comile setting
CC = mpicc
CFLAGS = -Wall

# target file
TARGET = test

# source files
SOURCES = main.c functions.c 
OBJECTS = $(SOURCES:.c=.o)

# make rule
all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# clean rule
clean:
	rm -f $(OBJECTS) $(TARGET)