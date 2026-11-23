import itertools
import json
import os
from typing import Any
from unittest.mock import MagicMock, patch

from crewai_tools.tools.wavespeed_tool import (
    WaveSpeedImageTool,
    WaveSpeedPredictionError,
    WaveSpeedTimeoutError,
    WaveSpeedVideoTool,
)
import pytest
import requests


BASE = "crewai_tools.tools.wavespeed_tool.wavespeed_base_tool"


def _response(
    body: Any, status_code: int = 200, headers: dict[str, str] | None = None
) -> MagicMock:
    response = MagicMock(spec=requests.Response)
    response.status_code = status_code
    response.headers = headers or {}
    response.ok = status_code < 400
    response.json.return_value = body
    response.text = json.dumps(body)
    return response


def _envelope(data: dict[str, Any]) -> MagicMock:
    return _response({"code": 200, "message": "success", "data": data})


@pytest.fixture(autouse=True)
def api_key():
    with patch.dict(os.environ, {"WAVESPEED_API_KEY": "test-key"}):
        yield


@pytest.fixture(autouse=True)
def no_sleep():
    with patch(f"{BASE}.time.sleep") as sleep:
        yield sleep


@pytest.fixture
def mock_post():
    with patch(f"{BASE}.requests.post") as post:
        post.return_value = _envelope({"id": "pred-1", "status": "created"})
        yield post


@pytest.fixture
def mock_get():
    with patch(f"{BASE}.requests.get") as get:
        yield get


def test_image_tool_text_to_image(mock_post, mock_get):
    mock_get.side_effect = [
        _envelope({"id": "pred-1", "status": "processing", "outputs": []}),
        _envelope(
            {
                "id": "pred-1",
                "status": "completed",
                "outputs": ["https://example.com/out.png"],
            }
        ),
    ]

    result = json.loads(
        WaveSpeedImageTool(resolution="1k").run(prompt="a red fox in the snow")
    )

    assert result == {
        "image_urls": ["https://example.com/out.png"],
        "model": "bytedance/seedream-v5.5",
        "prediction_id": "pred-1",
    }
    mock_post.assert_called_once()
    assert (
        mock_post.call_args.args[0]
        == "https://api.wavespeed.ai/api/v3/bytedance/seedream-v5.5"
    )
    assert mock_post.call_args.kwargs["json"] == {
        "prompt": "a red fox in the snow",
        "resolution": "1k",
    }
    assert mock_post.call_args.kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert (
        mock_get.call_args.args[0]
        == "https://api.wavespeed.ai/api/v3/predictions/pred-1/result"
    )


def test_image_tool_uses_edit_model_with_reference_images(mock_post, mock_get):
    mock_get.return_value = _envelope(
        {"id": "pred-1", "status": "completed", "outputs": ["https://e.com/1.png"]}
    )

    result = json.loads(
        WaveSpeedImageTool().run(
            prompt="put the cat on the sofa",
            images=["https://e.com/cat.png", "https://e.com/sofa.png"],
        )
    )

    assert result["model"] == "bytedance/seedream-v5.5/edit"
    assert mock_post.call_args.args[0].endswith("/bytedance/seedream-v5.5/edit")
    assert mock_post.call_args.kwargs["json"] == {
        "prompt": "put the cat on the sofa",
        "images": ["https://e.com/cat.png", "https://e.com/sofa.png"],
    }


def test_video_tool_text_and_image_to_video(mock_post, mock_get):
    mock_get.return_value = _envelope(
        {"id": "pred-1", "status": "completed", "outputs": ["https://e.com/v.mp4"]}
    )
    tool = WaveSpeedVideoTool(duration=5)

    text_result = json.loads(tool.run(prompt="waves at sunset"))
    assert text_result["video_url"] == "https://e.com/v.mp4"
    assert mock_post.call_args.args[0].endswith("/alibaba/wan-3.0/text-to-video")
    assert mock_post.call_args.kwargs["json"] == {
        "prompt": "waves at sunset",
        "duration": 5,
    }

    image_result = json.loads(
        tool.run(prompt="the dog runs", image="https://e.com/dog.png")
    )
    assert image_result["model"] == "alibaba/wan-3.0/image-to-video"
    assert mock_post.call_args.kwargs["json"] == {
        "prompt": "the dog runs",
        "image": "https://e.com/dog.png",
        "duration": 5,
    }


def test_extra_params_are_sent(mock_post, mock_get):
    mock_get.return_value = _envelope(
        {"id": "pred-1", "status": "completed", "outputs": ["https://e.com/1.png"]}
    )

    WaveSpeedImageTool(model="vendor/other-model", extra_params={"seed": 7}).run(
        prompt="a lighthouse"
    )

    assert mock_post.call_args.args[0].endswith("/vendor/other-model")
    assert mock_post.call_args.kwargs["json"] == {"seed": 7, "prompt": "a lighthouse"}


def test_envelope_error_raises_without_polling(mock_post, mock_get):
    mock_post.return_value = _response(
        {"code": 400, "message": "prompt is required", "data": None}
    )

    with pytest.raises(RuntimeError, match="prompt is required"):
        WaveSpeedImageTool().run(prompt="x")
    mock_get.assert_not_called()


def test_http_error_raises(mock_post, mock_get):
    mock_post.return_value = _response(
        {"code": 401, "message": "invalid api key"}, status_code=401
    )

    with pytest.raises(RuntimeError, match="HTTP 401: invalid api key"):
        WaveSpeedImageTool().run(prompt="x")
    mock_get.assert_not_called()


@pytest.mark.parametrize("status", ["failed", "cancelled", "timeout"])
def test_failed_prediction_raises(mock_post, mock_get, status):
    mock_get.return_value = _envelope(
        {"id": "pred-1", "status": status, "error": "content rejected"}
    )

    with pytest.raises(RuntimeError, match=f"pred-1 {status}: content rejected"):
        WaveSpeedImageTool().run(prompt="x")
    mock_post.assert_called_once()


def test_transient_poll_errors_are_retried_without_resubmitting(mock_post, mock_get):
    mock_get.side_effect = [
        requests.ConnectionError("reset"),
        _response({"message": "bad gateway"}, status_code=502),
        _envelope(
            {"id": "pred-1", "status": "completed", "outputs": ["https://e.com/1"]}
        ),
    ]

    result = json.loads(WaveSpeedImageTool().run(prompt="x"))

    assert result["image_urls"] == ["https://e.com/1"]
    mock_post.assert_called_once()
    assert mock_get.call_count == 3


def test_timeout_raises(mock_post, mock_get):
    mock_get.return_value = _envelope({"id": "pred-1", "status": "processing"})
    clock = itertools.count(0.0, 6.0)

    with patch(f"{BASE}.time.monotonic", side_effect=lambda: next(clock)):
        with pytest.raises(TimeoutError, match="pred-1 did not finish within 10"):
            WaveSpeedImageTool(timeout=10).run(prompt="x")
    mock_post.assert_called_once()


def test_missing_api_key_raises_before_request(mock_post):
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="WAVESPEED_API_KEY"):
            WaveSpeedImageTool().run(prompt="x")
    mock_post.assert_not_called()


# The same retry and timeout cases are covered for the Langflow WaveSpeed bundle:
# outputs that are not a list, 429/5xx while polling, and a timeout once a run id exists.
@pytest.mark.parametrize("outputs", ["https://e.com/1", {"url": "https://e.com/1"}, None])
def test_outputs_that_are_not_a_list_raise(mock_post, mock_get, outputs):
    mock_get.return_value = _envelope(
        {"id": "pred-1", "status": "completed", "outputs": outputs}
    )

    with pytest.raises(WaveSpeedPredictionError, match="outputs are not a list") as err:
        WaveSpeedImageTool().run(prompt="x")
    assert err.value.prediction_id == "pred-1"


def test_poll_429_and_5xx_back_off_and_honour_retry_after(mock_post, mock_get, no_sleep):
    mock_get.side_effect = [
        _response({"message": "slow down"}, status_code=429, headers={"Retry-After": "7"}),
        _response({"message": "bad gateway"}, status_code=502),
        _response({"message": "busy"}, status_code=503),
        _envelope({"id": "pred-1", "status": "completed", "outputs": ["https://e.com/1"]}),
    ]

    result = json.loads(WaveSpeedImageTool(poll_interval=1.0).run(prompt="x"))

    assert result["image_urls"] == ["https://e.com/1"]
    mock_post.assert_called_once()
    assert mock_get.call_count == 4
    # first poll waits the interval, then Retry-After, then doubling backoff
    assert [c.args[0] for c in no_sleep.call_args_list] == [1.0, 7.0, 14.0, 28.0]


def test_backoff_is_capped(mock_post, mock_get, no_sleep):
    mock_get.side_effect = [
        _response({}, status_code=500),
        _response({}, status_code=500),
        _envelope({"id": "pred-1", "status": "completed", "outputs": ["https://e.com/1"]}),
    ]

    WaveSpeedImageTool(poll_interval=20.0).run(prompt="x")

    assert [c.args[0] for c in no_sleep.call_args_list] == [20.0, 30.0, 30.0]


def test_poll_401_is_not_retried_and_keeps_the_run_id(mock_post, mock_get):
    mock_get.return_value = _response({"message": "unauthorized"}, status_code=401)

    with pytest.raises(WaveSpeedPredictionError, match="HTTP 401") as err:
        WaveSpeedImageTool().run(prompt="x")
    assert err.value.prediction_id == "pred-1"
    assert mock_get.call_count == 1


def test_timeout_keeps_the_run_id_and_points_to_the_result(mock_post, mock_get):
    mock_get.return_value = _envelope({"id": "pred-1", "status": "processing"})
    clock = itertools.count(0.0, 6.0)

    with patch(f"{BASE}.time.monotonic", side_effect=lambda: next(clock)):
        with pytest.raises(WaveSpeedTimeoutError) as err:
            WaveSpeedImageTool(timeout=10).run(prompt="x")
    assert err.value.prediction_id == "pred-1"
    assert "/api/v3/predictions/pred-1/result" in str(err.value)
    mock_post.assert_called_once()


def test_submit_timeout_is_not_retried_and_says_outcome_is_unknown(mock_post, mock_get):
    mock_post.side_effect = requests.ReadTimeout("slow")

    with pytest.raises(RuntimeError, match="may still have been created"):
        WaveSpeedImageTool().run(prompt="x")
    mock_post.assert_called_once()
    mock_get.assert_not_called()


def test_submit_5xx_is_not_retried_and_says_outcome_is_unknown(mock_post, mock_get):
    mock_post.return_value = _response({"message": "bad gateway"}, status_code=502)

    with pytest.raises(RuntimeError, match="may still have been created"):
        WaveSpeedImageTool().run(prompt="x")
    mock_post.assert_called_once()
    mock_get.assert_not_called()


def test_submit_that_never_connected_is_reported_as_unreachable(mock_post, mock_get):
    mock_post.side_effect = requests.ConnectTimeout("no route")

    with pytest.raises(RuntimeError, match="Could not reach the WaveSpeed API"):
        WaveSpeedImageTool().run(prompt="x")
    mock_post.assert_called_once()


def test_overall_timeout_also_limits_the_submission(mock_post, mock_get):
    mock_get.return_value = _envelope(
        {"id": "pred-1", "status": "completed", "outputs": ["https://e.com/1"]}
    )

    WaveSpeedImageTool(timeout=5, request_timeout=30).run(prompt="x")

    assert mock_post.call_args.kwargs["timeout"] == 5
