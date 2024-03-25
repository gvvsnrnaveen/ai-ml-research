INCLUDES += -I../utils
OBJS += ../utils/nav_metrics.o
all:
	$(CXX) decision_tree_training.cc -o decision_tree_training $(OBJS) $(INCLUDES) -std=c++11 -larmadillo -lmlpack -lgomp -fopenmp -lboost_serialization

clean:
	rm -rf decision_tree_training
