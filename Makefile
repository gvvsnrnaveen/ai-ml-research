all:
	make -C utils
	make -C decision_tree

clean:
	make -C utils clean
	make -C decision_tree clean
