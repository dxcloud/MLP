CC = g++
CFLAGS = -Wall -Werror -O
EXEC_NAME = networking
OBJ_FILES := main.o network.o

all : $(EXEC_NAME)

clean :
	rm -rf $(EXEC_NAME) $(OBJ_FILES)
	
$(EXEC_NAME) : $(OBJ_FILES)
	$(CC) -o $(EXEC_NAME) $(OBJ_FILES)
	 
$(OBJ_FILES):%.o : %.cpp
	$(CC) $(CFLAGS) -MMD -o $@ -c $<
