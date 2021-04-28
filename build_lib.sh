#!/bin/bash
cd post_process_lib
make -f Makefile.pcie clean
make -f Makefile.pcie
#cp libPostProcess.so ../
cd ..
