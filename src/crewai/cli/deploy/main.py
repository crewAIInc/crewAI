import json
import os
from os import getenv

import requests
from rich import print

from .utils import fetch_and_json_env_file, get_git_remote_url, get_project_name


def get_auth_token():
    return os.environ.get(
        "TOKEN", "958303356b9a21884a83ddb6e774cc06c6f1dd0e04222fbc5a4e8a9ae02c140e"
    )


class DeployCommand:
    def __init__(self):
        # self.base_url = os.environ.get("BASE_URL", "https://www.crewai.com/api")
        self.base_url = getenv("BASE_URL", "http://localhost:3000/crewai_plus/api")

        self.project_name = get_project_name()
        self.remote_repo_url = get_git_remote_url()

    def deploy(self):
        print("Deploying the crew...")
        response = requests.post(
            f"{self.base_url}/crews/by-name/{self.project_name}/deploy",
            headers={"Authorization": f"Bearer {get_auth_token()}"},
        )
        print(response.json())

    def create_crew(self):
        print("Creating deployment...")
        env_vars = fetch_and_json_env_file()
        payload = {
            "deploy": {
                "name": self.project_name,
                "repo_clone_url": self.remote_repo_url,
                "env": env_vars,
            }
        }
        response = requests.post(
            f"{self.base_url}/crews",
            data=json.dumps(payload),
            headers={
                "Authorization": f"Bearer {get_auth_token()}",
                "Content-type": "application/json",
            },
        )
        print(response.json())

    def list_crews(self):
        print("Listing all Crews")

        response = requests.get(
            f"{self.base_url}/crews",
            headers={"Authorization": f"Bearer {get_auth_token()}"},
        )
        crews_data = response.json()
        if response.status_code == 200:
            print()
            if len(crews_data):
                for crew_data in crews_data:
                    print(
                        f"- {crew_data['name']} ({crew_data['uuid']}) [blue]\[{crew_data['status']}]"
                    )
            else:
                print("You don't have any crews yet. Let's create one!")
                print()
                print("\t[green]crewai create --name \[name]")

    def get_crew_status(self):
        print("Getting deployment status...")
        response = requests.get(
            f"{self.base_url}/crews/by-name/{self.project_name}/status",
            headers={"Authorization": f"Bearer {get_auth_token()}"},
        )
        if response.status_code == 200:
            status_data = response.json()
            print(status_data)
            print("Name:\t", status_data["name"])
            print("Status:\t", status_data["status"])
            print()
            print("usage:")
            print(f"\tcrewai inputs --name \"{status_data['name']}\" ")
            print(
                f"\tcrewai kickoff --name \"{status_data['name']}\" --inputs [INPUTS]"
            )
        else:
            print(response.json())

    def get_crew_logs(self, log_type="deployment"):
        print("Getting deployment logs...")
        response = requests.get(
            f"{self.base_url}/crews/by-name/{self.project_name}/logs/{log_type}",
            headers={"Authorization": f"Bearer {get_auth_token()}"},
        )

        if response.status_code == 200:
            log_messages = response.json()
            for log_message in log_messages:
                print(
                    f"{log_message['timestamp']} - {log_message['level']}: {log_message['message']}"
                )
        else:
            print(response.text)

    def remove_crew(self):
        print("Removing deployment...")
        response = requests.delete(
            f"{self.base_url}/crews/by-name/{self.project_name}",
            headers={"Authorization": f"Bearer {get_auth_token()}"},
        )
        if response.status_code == 204:
            print(f"Crew '{self.project_name}' removed successfully.")

    def signup(self):
        print("Signing Up")
        response = requests.get(f"{self.base_url}/signup_link")
        if response.status_code == 200:
            token = response.json()["token"]
            signup_link = response.json()["signup_link"]
            print("Temporary credentials: ", token)
            print(
                "We are trying to open the following signup link below.\n"
                "If it doesn't open to you, copy-and-paste into our browser to procceed."
            )
            print()
            print(signup_link)
            # webbrowser.open(signup_link, new=2)
        else:
            print(response.text)
