
import os
import time
import requests
from typing import Type, Any
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

class TwoCaptchaSolverSchema(BaseModel):
    website_url: str = Field(..., description="URL ???????? ? ??????.")
    sitekey: str = Field(..., description="???? sitekey ?? HTML ???? ????????.")
    captcha_type: str = Field(default="RecaptchaV2TaskProxyless", description="??? ?????.")

class TwoCaptchaSolverTool(BaseTool):
    name: str = "twocaptcha_solver"
    description: str = "?????? ????? (reCAPTCHA, hCaptcha, Turnstile) ????? ?????? 2Captcha."
    args_schema: Type[BaseModel] = TwoCaptchaSolverSchema
    api_key: str = Field(default_factory=lambda: os.getenv("TWOCAPTCHA_API_KEY", ""))

    def _run(self, website_url: str, sitekey: str, captcha_type: str = "RecaptchaV2TaskProxyless", **kwargs: Any) -> str:
        api_key = self.api_key or os.getenv("TWOCAPTCHA_API_KEY")
        if not api_key: return "??????: API ???? 2Captcha ?? ??????."
        payload = {"clientKey": api_key, "task": {"type": captcha_type, "websiteURL": website_url, "websiteKey": sitekey}}
        try:
            res = requests.post("https://api.2captcha.com/createTask", json=payload).json()
            if res.get("errorId") != 0: return f"?????? API: {res.get('errorDescription')}"
            task_id = res.get("taskId")
            for _ in range(20):
                time.sleep(5)
                result = requests.post("https://api.2captcha.com/getTaskResult", json={"clientKey": api_key, "taskId": task_id}).json()
                if result.get("status") == "ready":
                    sol = result.get("solution", {})
                    return f"??????! ?????: {sol.get('gRecaptchaResponse') or sol.get('token')}"
            return "??????? ??????? ?????."
        except Exception as e: return f"??????: {str(e)}"

