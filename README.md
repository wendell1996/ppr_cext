# ppr_cext

## Linux安装
1. `make dlib`

2. `pip install - r requires.txt`

3. `ln -s "$(pwd)"/libppr.so ${你的lib库地址}/libppr.so`

或者`cp`
比如`cp libppr.so /usr/lib`

或者copy到conda环境下面`cp libppr.so ${你的anaconda目录}/envs/${使用环境名}/lib`

4. `python setup.py install --record logs`

## Mac安装
同Linux安装

## 一键简易安装(从源文件安装)
`source install.sh`

若运行多次，`~/.bashrc` 下可能有冗余的，建议按以下操作安装

## 安装步骤（从源文件安装）
要求python3

1.`pip install -r requires.txt` 安装`numpy、scipy、Cython`

2.`make dlib` 生成动态库

3.`ln -s "$(pwd)"/libppr.so ${你的lib库地址}/libppr.so`
建立动态库软连接方便修改动态库

4.`python setup.py build_ext --inplace && python setup.py install
--record logs`

卸载

1.`cat logs | xargs rm`

