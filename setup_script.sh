#!/bin/bash

# Default values
USERNAME=""
PASSWORD=""

# Parse flags
while getopts u:p: flag
do
    case "${flag}" in
        u) USERNAME=${OPTARG};;
        p) PASSWORD=${OPTARG};;
    esac
done

sudo apt update && sudo apt install -y
sudo apt install -y python3-pip
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker

if [ -d "pallscripting" ]; then
    rm -rf pallscripting
fi

git clone https://github.com/CodeHalwell/pallscripting.git

cd pallscripting

touch .env

mkdir -p .streamlit
touch .streamlit/secrets.toml

# Write username and password to secrets.toml
echo "[secrets]" > .streamlit/secrets.toml
echo "username = \"$USERNAME\"" >> .streamlit/secrets.toml
echo "password = \"$PASSWORD\"" >> .streamlit/secrets.toml