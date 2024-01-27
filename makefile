
CC = gcc
RM = rm -f

OBJS = main.o direct.o problem.o
OBJSLIB = direct.o

FFLAGS = -fPIC --static --shared -O3
#FFLAGS = -g


all: direct lib

direct: $(OBJS) headers_direct.h
	$(CC) -o direct $(OBJS) -lm	

lib:  $(OBJSLIB)
	$(CC) -fPIC --shared -lm -o libdirect.a $(OBJSLIB)

.SUFFIXES : .c   .o

.c.o:   $* ; $(CC) $(FFLAGS) -c $*.c

clean: 
	$(RM) *.o
	$(RM) direct
	$(RM) libdirect.a

