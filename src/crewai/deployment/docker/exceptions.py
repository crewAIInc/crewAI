class DockerError(Exception):
    """Base exception for Docker-related errors in CrewAI deployments."""
    pass


class DockerBuildError(DockerError):
    """Exception raised when Docker build fails."""
    pass


class DockerRunError(DockerError):
    """Exception raised when Docker container fails to run."""
    pass


class DockerComposeError(DockerError):
    """Exception raised when docker-compose commands fail."""
    pass
