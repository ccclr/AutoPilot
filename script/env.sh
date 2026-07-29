sudo apt-get update
sudo apt-get install tmux

curl https://sh.rustup.rs -sSf | sh -s -- -y
sudo apt-get install libclang-dev

sudo apt-get update
sudo apt-get install iproute2

sudo apt-get install python3-pip

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pip install -r "${SCRIPT_DIR}/requirements.txt"