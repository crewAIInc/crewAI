import os
import time
import requests
from typing import Type, Any, Optional
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

class TwoCaptchaSolverSchema(BaseModel):
    """Input for TwoCaptchaSolverTool."""
    website_url: str = Field(..., description="The full URL of the website where the captcha is located.")
    sitekey: str = Field(..., description="The 'data-sitekey' value found in the website's HTML.")
    captcha_type: str = Field(
        default="RecaptchaV2TaskProxyless", 
        description="Type of captcha to solve (e.g., RecaptchaV2TaskProxyless, hCaptchaTaskProxyless, TurnstileTaskProxyless)."
    )

class TwoCaptchaSolverTool(BaseTool):
    name: str = "twocaptcha_solver"
    description: str = (
        "Automated tool for solving various captchas (reCAPTCHA, hCaptcha, Cloudflare Turnstile) "
        "via the 2Captcha service. Requires the TWOCAPTCHA_API_KEY environment variable."
    )
    args_schema: Type[BaseModel] = TwoCaptchaSolverSchema
    api_key: Optional[str] = Field(default=None)

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key or os.getenv("TWOCAPTCHA_API_KEY")

    def _run(
        self, 
        website_url: str, 
        sitekey: str, 
        captcha_type: str = "RecaptchaV2TaskProxyless", 
        **kwargs: Any
    ) -> str:
        if not self.api_key:
            return "Error: 2Captcha API key not found. Please set the TWOCAPTCHA_API_KEY environment variable."

        create_task_url = "https://api.2captcha.com/createTask"
        get_result_url = "https://api.2captcha.com/getTaskResult"

        payload = {
            "clientKey": self.api_key,
            "task": {
                "type": captcha_type,
                "websiteURL": website_url,
                "websiteKey": sitekey
            }
        }

        try:
            response = requests.post(create_task_url, json=payload, timeout=20)
            response.raise_for_status()
            data = response.json()

            if data.get("errorId") != 0:
                return f"2Captcha API Error: {data.get('errorDescription')}"

            task_id = data.get("taskId")
            
            # Polling for the result (max 2 minutes)
            for _ in range(24):
                time.sleep(5)
                result_payload = {"clientKey": self.api_key, "taskId": task_id}
                result_resp = requests.post(get_result_url, json=result_payload, timeout=20)
                result_resp.raise_for_status()
                result_data = result_resp.json()

                if result_data.get("status") == "ready":
                    solution = result_data.get("solution", {})
                    # Return gRecaptchaResponse for reCAPTCHA or token for other types
                    return solution.get("gRecaptchaResponse") or solution.get("token") or "Error: No token in response."
            
            return "Error: Timed out waiting for captcha solution."

        except requests.exceptions.RequestException as e:
            return f"Network error connecting to 2Captcha: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"