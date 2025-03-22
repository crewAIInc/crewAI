import os

import click
from rich.console import Console

from crewai.deployment.main import Deployment

console = Console()

@click.group()
def deploy():
    """CrewAI deployment tools for containerizing and running CrewAI workflows."""
    pass

@deploy.command("create")
@click.argument("config_path", type=click.Path(exists=True))
def create_deployment(config_path):
    """Create a new deployment from a configuration file."""
    try:
        console.print("Creating deployment...", style="bold blue")
        deployment = Deployment(config_path)
        deployment.prepare()
        console.print(f"Deployment prepared at {deployment.deployment_dir}", style="bold green")
        console.print(f"Configuration:", style="bold blue")
        console.print(f"  Name: {deployment.config.name}")
        console.print(f"  Port: {deployment.config.port}")
        console.print(f"  Host: {deployment.config.host}")
        console.print(f"  Crews: {[c.name for c in deployment.config.crews]}")
        console.print(f"  Flows: {[f.name for f in deployment.config.flows]}")
    except Exception as e:
        console.print(f"Error creating deployment: {e}", style="bold red")
        
@deploy.command("build")
@click.argument("deployment_name")
def build_deployment(deployment_name):
    """Build Docker image for deployment."""
    try:
        console.print("Building deployment...", style="bold blue")
        deployment_dir = f"./deployments/{deployment_name}"
        if not os.path.exists(deployment_dir):
            console.print(f"Deployment {deployment_name} not found", style="bold red")
            return
            
        config_path = f"{deployment_dir}/deployment_config.json"
        deployment = Deployment(config_path)
        deployment.build()
        console.print("Build completed successfully", style="bold green")
    except Exception as e:
        console.print(f"Error building deployment: {e}", style="bold red")
        
@deploy.command("start")
@click.argument("deployment_name")
def start_deployment(deployment_name):
    """Start a deployment."""
    try:
        console.print("Starting deployment...", style="bold blue")
        deployment_dir = f"./deployments/{deployment_name}"
        if not os.path.exists(deployment_dir):
            console.print(f"Deployment {deployment_name} not found", style="bold red")
            return
            
        config_path = f"{deployment_dir}/deployment_config.json"
        deployment = Deployment(config_path)
        deployment.start()
        console.print(f"Deployment {deployment_name} started", style="bold green")
        console.print(f"API server running at http://{deployment.config.host}:{deployment.config.port}")
    except Exception as e:
        console.print(f"Error starting deployment: {e}", style="bold red")
        
@deploy.command("stop")
@click.argument("deployment_name")
def stop_deployment(deployment_name):
    """Stop a deployment."""
    try:
        console.print("Stopping deployment...", style="bold blue")
        deployment_dir = f"./deployments/{deployment_name}"
        if not os.path.exists(deployment_dir):
            console.print(f"Deployment {deployment_name} not found", style="bold red")
            return
            
        config_path = f"{deployment_dir}/deployment_config.json"
        deployment = Deployment(config_path)
        deployment.stop()
        console.print(f"Deployment {deployment_name} stopped", style="bold green")
    except Exception as e:
        console.print(f"Error stopping deployment: {e}", style="bold red")
        
@deploy.command("logs")
@click.argument("deployment_name")
def show_logs(deployment_name):
    """Show logs for a deployment."""
    try:
        console.print("Fetching logs...", style="bold blue")
        deployment_dir = f"./deployments/{deployment_name}"
        if not os.path.exists(deployment_dir):
            console.print(f"Deployment {deployment_name} not found", style="bold red")
            return
            
        config_path = f"{deployment_dir}/deployment_config.json"
        deployment = Deployment(config_path)
        deployment.logs()
    except Exception as e:
        console.print(f"Error fetching logs: {e}", style="bold red")
