pip install git+https://github.com/ucbrise/clipper.git@develop#subdirectory=clipper_admin

# Uninstall if installed in case of conflict 
pip uninstall clipper-admin

pip install -r requirements.txt

git clone https://github.com/thtrieu/darkflow.git

cd darkflow 

python setup.py build_ext --inplace

cd ..