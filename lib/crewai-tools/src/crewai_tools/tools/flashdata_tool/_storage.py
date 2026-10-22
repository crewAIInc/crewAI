"""Customer storage validation. Never include credentials in errors."""

import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import parse_qs

from jsonschema import Draft7Validator

from crewai_tools.tools.flashdata_tool._client import FlashDataError


STORAGE_SCHEMA = json.loads(Path(__file__).with_name("storage.schema.json").read_text())


def storage_config(raw: str) -> dict[str, Any]:
    try:
        storage = json.loads(raw)
    except (ValueError, TypeError):
        raise FlashDataError("Download storage must contain a JSON object.") from None
    if not isinstance(storage, dict) or not Draft7Validator(STORAGE_SCHEMA).is_valid(
        storage
    ):
        raise FlashDataError(
            "Invalid download storage credential. Check its fields and types."
        )
    kind = storage["storage_type"]
    bucket = storage["storage_url"].strip()
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", bucket) or ".." in bucket:
        raise FlashDataError(
            "storage_url must be a bucket/container name; put folders in object_prefix."
        )
    if kind == "s3" and (
        re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", bucket)
        or ".-" in bucket
        or "-." in bucket
    ):
        raise FlashDataError("storage_url must be a valid AWS S3 bucket name.")
    storage["storage_url"] = bucket
    provider_fields = (
        ["service_account"]
        if kind == "gcs"
        else ["account_name", "sas_token"]
        if kind == "azure"
        else ["access_key", "secret_key", "region"]
        if kind == "cos"
        else ["access_key", "secret_key", "region", "endpoint_url"]
    )
    if set(storage) - {
        "storage_type",
        "storage_url",
        "object_prefix",
        *provider_fields,
    }:
        raise FlashDataError(
            "The storage credential contains fields for a different provider."
        )
    if "object_prefix" in storage:
        prefix = storage["object_prefix"].strip().removesuffix("/")
        if (
            not prefix
            or len(prefix) > 512
            or any(
                part in (".", "..") or not re.fullmatch(r"[A-Za-z0-9._-]+", part)
                for part in prefix.split("/")
            )
        ):
            raise FlashDataError(
                "object_prefix must use safe folder names, up to 512 characters."
            )
        storage["object_prefix"] = prefix
    if kind == "gcs":
        account = storage.get("service_account", {})
        if (
            not all(
                account.get(k) for k in ("project_id", "client_email", "private_key")
            )
            or not re.fullmatch(
                r"[^@\s]+@[^@\s]+\.gserviceaccount\.com",
                account.get("client_email", ""),
            )
            or "BEGIN PRIVATE KEY" not in account.get("private_key", "")
        ):
            raise FlashDataError(
                "GCS requires project_id, service-account client_email and a PEM private_key."
            )
        storage["service_account"] = {
            key: account[key] for key in ("project_id", "client_email", "private_key")
        }
    elif kind == "azure":
        sas = storage.get("sas_token", "").removeprefix("?")
        fields = parse_qs(sas)
        if (
            not storage.get("account_name")
            or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", bucket)
            or not all(fields.get(k) for k in ("sig", "se", "sv"))
            or "w" not in fields.get("sp", [""])[0]
            or fields.get("sr") != ["c"]
            or "\n" in sas
            or "\r" in sas
            or len(sas) > 4096
        ):
            raise FlashDataError(
                "Azure requires an account name and container SAS with write permission and expiry."
            )
        storage["sas_token"] = sas
    else:
        if not all(storage.get(k, "").strip() for k in ("access_key", "secret_key")):
            raise FlashDataError(
                "Download storage requires access_key and secret_key in the provider credential."
            )
        if kind == "s3_compatible" and not storage.get("endpoint_url"):
            raise FlashDataError("S3-compatible storage requires endpoint_url.")
        region = storage.get("region", "")
        if (
            (kind in ("cos", "bos", "tos") and "." in bucket)
            or (
                kind == "cos"
                and (
                    not re.fullmatch(r"ap-[a-z0-9-]+", region)
                    or not re.search(r"-\d{5,}$", bucket)
                )
            )
            or (kind == "bos" and region not in ("bj", "su", "hkg"))
            or (
                kind == "tos"
                and not re.fullmatch(r"(?:cn|ap|eu|us)-[a-z0-9-]+", region)
            )
        ):
            raise FlashDataError("Check the storage provider region and bucket name.")
    return storage
