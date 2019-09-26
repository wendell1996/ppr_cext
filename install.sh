#!/bin/bash

pip install -r requires.txt
cur_dir="$(pwd)"
make dlib
echo export LD_LIBRARY_PATH=${cur_dir}:$LD_LIBRARY_PATH >> ~/.bashrc
export LD_LIBRARY_PATH=${cur_dir}:$LD_LIBRARY_PATH
# or 
# ln -s ${cur_dir}/libppr.so ${your c++ lib dir}/libppr.so
# example "ln -s /home/chenhuidi/ant/ppr_cext/libppr.so /home/chenhuidi/anaconda3/envs/py3.6/lib/libppr.so"
python setup.py build_ext --inplace &&\
python setup.py install --record logs
