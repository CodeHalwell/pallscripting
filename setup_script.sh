# create the .streamlit directory and the .env file first before running this script

sudo apt update && sudo apt install -y;
sudo apt install -y python3-pip;
python3 -m venv .venv;
source .venv/bin/activate;
pip install -r requirements.txt;
curl -fsSL https://get.docker.com -o get-docker.sh;
sudo sh get-docker.sh;
sudo usermod -aG docker ubuntu;
newgrp docker;

git clone https://github.com/CodeHalwell/pallscripting.git;

cd pallscripting;

docker build -t halwelld88/pallscripting .;

docker run -d -p 8501:8501 halwelld88/pallscripting