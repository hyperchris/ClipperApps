sudo apt-get install python-tk

# Uninstall if installed in case of conflict 
sudo pip uninstall clipper-admin

pip install -r requirements.txt

git clone https://github.com/thtrieu/darkflow.git

cd darkflow 

python setup.py build_ext --inplace

cd ..