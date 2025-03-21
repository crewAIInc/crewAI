import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

class DockerContainer:
    """
    Manages Docker containers for CrewAI deployments.
    """
    def __init__(self, deployment_dir: str, name: str):
        self.deployment_dir = Path(deployment_dir)
        self.name = name
        self.dockerfile_path = self.deployment_dir / "Dockerfile"
        self.compose_path = self.deployment_dir / "docker-compose.yml"
        
    def generate_dockerfile(self, requirements: List[str] = None):
        """Generate a Dockerfile for the deployment."""
        template_dir = Path(__file__).parent / "templates"
        dockerfile_template = template_dir / "Dockerfile"
        
        os.makedirs(self.deployment_dir, exist_ok=True)
        shutil.copy(dockerfile_template, self.dockerfile_path)
        
        # Add requirements if specified
        if requirements:
            with open(self.dockerfile_path, "a") as f:
                f.write("\n# Additional requirements\n")
                f.write(f"RUN pip install {' '.join(requirements)}\n")
                
    def generate_compose_file(self, port: int = 8000):
        """Generate a docker-compose.yml file for the deployment."""
        template_dir = Path(__file__).parent / "templates"
        compose_template = template_dir / "docker-compose.yml"
        
        # Read template and replace placeholders
        with open(compose_template, "r") as f:
            template = f.read()
            
        compose_content = template.replace("{{name}}", self.name)
        compose_content = compose_content.replace("{{port}}", str(port))
        
        with open(self.compose_path, "w") as f:
            f.write(compose_content)
            
    def build(self):
        """Build the Docker image."""
        cmd = ["docker", "build", "-t", f"crewai-{self.name}", "."]
        subprocess.run(cmd, check=True, cwd=self.deployment_dir)
        
    def start(self):
        """Start the Docker containers using docker-compose."""
        cmd = ["docker-compose", "up", "-d"]
        subprocess.run(cmd, check=True, cwd=self.deployment_dir)
        
    def stop(self):
        """Stop the Docker containers."""
        cmd = ["docker-compose", "down"]
        subprocess.run(cmd, check=True, cwd=self.deployment_dir)
        
    def logs(self):
        """Get container logs."""
        cmd = ["docker-compose", "logs"]
        subprocess.run(cmd, check=True, cwd=self.deployment_dir)
