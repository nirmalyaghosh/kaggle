#!/bin/bash

################################################
# Provisions the VM for this Kaggle competition
#
# Credits :
# https://github.com/nirmalyaghosh/kaggle-vm
################################################

function mssg {
    now=$(date +"%T")
    echo "[$now] $1"
    shift
}

mssg "Provisioning the VM for this Kaggle competition ..."

mssg "Updating the package index files. Usually takes ~ 6 minutes, depending on the speed of your network ..."
apt-get -y update >/dev/null 2>&1

################################################
# apt-fast
mssg "Installing apt-fast to try speed things up ..."
apt-get install -y aria2 --no-install-recommends >/dev/null 2>&1
aptfast=apt-fast
if [[ ! -f $aptfast ]]; then
    wget https://raw.githubusercontent.com/ilikenwf/apt-fast/master/apt-fast >/dev/null 2>&1
    wget https://raw.githubusercontent.com/ilikenwf/apt-fast/master/apt-fast.conf >/dev/null 2>&1
    cp apt-fast /usr/bin/
    chmod +x /usr/bin/apt-fast
    cp apt-fast.conf /etc
fi

mssg "Installing pip ..."
apt-fast -y install python-pip >/dev/null 2>&1
pip install --upgrade pip
mssg "Installing Git"
apt-fast install git -y > /dev/null 2>&1
mssg "Installing python-dev "
apt-fast install -y python-dev >/dev/null 2>&1

################################################
# Miniconda
mssg "Downloading & Installing Miniconda ..."
miniconda=Miniconda3-4.0.5-Linux-x86_64.sh
if [[ ! -f $miniconda ]]; then
    wget --quiet http://repo.continuum.io/miniconda/$miniconda
    chmod +x $miniconda
    ./$miniconda -b -p /home/vagrant/miniconda
    echo 'export PATH="/home/vagrant/miniconda/bin:$PATH"' >> /home/vagrant/.bashrc
    source /home/vagrant/.bashrc
    chown -R vagrant:vagrant /home/vagrant/miniconda
    /home/vagrant/miniconda/bin/conda install conda-build anaconda-client anaconda-build -y -q
fi

################################################
# Install the essential Python packages : pandas, scikit-learn, xgboost, etc.
mssg "Installing pandas ..."
/home/vagrant/miniconda/bin/conda install pandas -y -q
mssg "Installing scikit-learn ..."
/home/vagrant/miniconda/bin/conda install scikit-learn -y -q
mssg "Installing imbalanced-learn ..."
/home/vagrant/miniconda/bin/conda install -c glemaitre imbalanced-learn -y -q
mssg "Upgrading pip ..."
/home/vagrant/miniconda/bin/pip install --upgrade pip
mssg "Installing XGBoost ..."
/home/vagrant/miniconda/bin/pip install xgboost==0.6a2
mssg "Installing other packages ..."
/home/vagrant/miniconda/bin/pip install -r /home/vagrant/requirements.txt

################################################
mssg "Installing IPython Notebook server"
mkdir -p /home/vagrant/notebooks
chown -R vagrant:vagrant /home/vagrant/notebooks
/home/vagrant/miniconda/bin/pip install ipython[notebook]
