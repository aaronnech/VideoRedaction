CC=g++
CVFLAGS=`pkg-config --cflags opencv`
CVLIBS=`pkg-config --libs opencv`
BOOSTFLAGS=-lboost_thread


executable:
	$(CC) main.cc -o executable $(BOOSTFLAGS) $(CVLIBS) $(CVFLAGS)

clean:
	rm executable
