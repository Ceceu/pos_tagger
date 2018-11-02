# Update the apt package index:
sudo apt-get update

# Install packages to allow apt to use a repository over HTTPS:
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common

# Add Docker’s official GPG key:
 curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

 # Set up the stable repository
 sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"

# Update the apt package index:
sudo apt-get update

# Install the latest version of Docker CE, or go to the next step to install a specific version:
sudo apt-get install docker-ce

# Create the docker group.
sudo groupadd docker

# Add your user to the docker group.
sudo usermod -aG docker $USER