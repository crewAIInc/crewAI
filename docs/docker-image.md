
# Running Docker Container with GPU Support and Mounting

This README provides instructions on how to run the `jimjim12/crewai-ollama:latest` Docker image with GPU support and mounting a host directory.

## Prerequisites

- **Docker Installed**: Docker must be installed on your machine.

- **NVIDIA GPU and Drivers**: If using GPU, ensure you have the appropriate NVIDIA drivers installed.

- **NVIDIA Docker Toolkit**: Install the NVIDIA Container Toolkit for enabling GPU access in Docker containers.

Add GPG Key
`curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -`

Add the NVIDIA GPU Cloud (NGC) container repository:
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
```

Refresh the package list:
`sudo apt-get update`

Upgrade packages:
`sudo apt-get upgrade -y`

Install the NVIDIA Container Toolkit:
`sudo apt-get install -y nvidia-docker2`

Restart the Docker service:
`sudo systemctl restart docker`

## Steps to Run the Docker Container

### Pull the Docker Image
First, pull the image from Docker Hub:
```bash
docker pull jimjim12/crewai-ollama:latest
```

### Determine the Mount Path
Decide the directory on your host machine you want to mount into the Docker container. For example, `/home/user/data` from your host to `/data` inside the container.
- You can mount multiple volumes by simply adding another `-v /source/data:/destination/directory` flag.
- There are no limits to the amount of volumes that can be added.
- A good recommendation is to add a volume where you store your locally downloaded llms, a volume to access any scripts you want to run, and a volume that acts as an isolated directory to run scripts.

### Run the Container with GPU Support and Mounting
Run the container using the `docker run` command with the `--gpus` flag for GPU support, `-p` to bind the host port to the container port, and `-v` for volume mounting:
```bash
docker run --name NameYourContainer -p 11434:11434 -it --gpus all -v /home/user/data:/data jimjim12/crewai-ollama:latest
```
- `--gpus all` allows access to all available GPUs.
- `-v /home/user/data:/data` mounts the `/home/user/data` directory from your host to `/data` inside the container. This is optional but recommended.
- Replace `/home/user/data` with your actual directory path.
- `-it` Keeps standard input (STDIN) open even if not attached. Allows you to interact with the cli. Optional.
- `-p` binds the host port to the container port. If not specified docker will assign a random port.

### Run with Specific GPU(s)
To specify particular GPU(s), replace `all` in the `--gpus` flag with the specific GPU IDs:
```bash
docker run --name NameYourContainer -p 11434:11434 -it --gpus '"device=0,1"' -v /home/user/data:/data jimjim12/crewai-ollama:latest
```
### Run the Container with only CPU support and Mounting
```bash
docker run --name NameYourContainer -p 11434:11434 -it -v /home/user/data:/data jimjim12/crewai-ollama:latest
```
-  If you don't want to use all CPUs, you can set `--cpus=<value>` flag. Learn more about this and other container restraints here: `https://docs.docker.com/config/containers/resource_constraints/`

## Additional Notes
- The directory to mount should have necessary permissions for Docker to access and modify files.
- Windows users can run from WSL or CMD. When mounting drives with CMD, replace `/home/user/data` with the Windows equivelant `C:\Users\User\Your\Data`
- Use `-d` to run the container in detached mode or `-it` for interactive mode (Keeps STDIN open to interact with the CLI).
-  This is a bare bones image. It has the CrewAI framework, Ollama, DuckDuckGoSearch, and the basic required packages to run a crew preinstalled. You may need to install other packages depending on what your crew uses. (i.e the langchain-expermimental package for the human input tool)

## Verification
Verify GPU utilization using `nvidia-smi` command after the container is running.