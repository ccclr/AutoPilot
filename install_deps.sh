sudo apt-get update
sudo apt-get -y upgrade
sudo apt-get -y autoremove

# The following dependencies prevent the error: [error: linker `cc` not found].
sudo apt-get -y install build-essential
sudo apt-get -y install cmake

# Install rust (non-interactive).
curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env
rustup default stable

# This is missing from the Rocksdb installer (needed for Rocksdb).
sudo apt-get install -y clang

# Tmux if not already installed
sudo apt-get install -y tmux

sudo apt-get remove python3-pip
wget https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py

echo 'export PATH=$PATH:~/.local/bin' >> ~/.bashrc
exec bash