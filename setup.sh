sudo apt-get install python-tk

pip install -r requirements.txt

pip install git+https://github.com/ucbrise/clipper.git@develop#subdirectory=clipper_admin

git clone https://github.com/thtrieu/darkflow.git

cd darkflow 

python setup.py build_ext --inplace

cd ..