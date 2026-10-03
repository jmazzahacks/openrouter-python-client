"""A paused consumer must have a checkpoint for exactly the events it received."""

import json
from unittest.mock import Mock

import pytest
from smartsurge.streaming import StreamingState

from openrouter_client.auth import AuthManager
from openrouter_client.http import HTTPManager
from openrouter_client.streaming import (
    StreamingChatCompletionsRequest,
    StreamingCompletionsRequest,
)


@pytest.mark.parametrize("chat", [False, True])
@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize(
    "separators",
    [
        (b"\n\n", b"\n\n"),
        (b"\r\n\r\n", b"\r\n\r\n"),
        (b"\n\n", b"\r\n\r\n"),
        (b"\r\n\r\n", b"\n\n"),
    ],
)
@pytest.mark.parametrize("fragmented", [False, True])
def test_checkpoint_exists_before_yield_and_excludes_unread_events(
    tmp_path, chat, resume, separators, fragmented
):
    def frame(text, separator):
        choice = {"delta": {"content": text}} if chat else {"text": text}
        return b"data: " + json.dumps({"choices": [choice]}).encode() + separator

    first = frame("Hello", separators[0])
    second = frame(" world", separators[1])
    wire = first + second + frame(" unread", separators[0])
    response = Mock(status_code=200)
    response.iter_content.return_value = (
        [wire[:13], wire[13:]] if fragmented else [wire]
    )
    http = Mock(spec=HTTPManager)
    http.client = Mock()
    http.client.post.return_value = response
    auth = AuthManager(api_key="test-key")
    state_file = tmp_path / "state.json"
    request_class = (
        StreamingChatCompletionsRequest if chat else StreamingCompletionsRequest
    )
    request = request_class(
        http_manager=http,
        auth_manager=auth,
        endpoint=(
            "https://openrouter.ai/api/v1/chat/completions"
            if chat
            else "https://openrouter.ai/api/v1/completions"
        ),
        headers=auth.get_auth_headers(),
        params={"model": "test/model"},
        state_file=str(state_file),
        **(
            {"messages": [{"role": "user", "content": "Hi"}]}
            if chat
            else {"prompt": "Hi"}
        ),
    )
    prefix = frame("Earlier ", separators[0]) if resume else b""
    if resume:
        request.accumulated_data.extend(prefix)
        request.position = len(prefix)
        request.save_state()
    stream = request.resume_stream() if resume else request.stream()
    try:
        first_result = next(stream)
        state = StreamingState.model_validate_json(state_file.read_text())
        assert state.accumulated_data == prefix + first
        assert state.last_position == len(prefix + first)
        assert "Authorization" not in state.headers
        assert request.get_result() == [first_result]
        if resume:
            sent = http.client.post.call_args.kwargs["json"]
            if chat:
                assert sent["messages"][-1] == {
                    "role": "assistant",
                    "content": "Earlier ",
                }
            else:
                assert sent["prompt"] == "HiEarlier "

        second_result = next(stream)
        state = StreamingState.model_validate_json(state_file.read_text())
        assert state.accumulated_data == prefix + first + second
        assert state.last_position == len(prefix + first + second)
        assert request.get_result() == [first_result, second_result]

        request.cancel()
        assert list(stream) == []
        assert StreamingState.model_validate_json(state_file.read_text()) == state
        assert request.get_result() == [first_result, second_result]
        response.close.assert_called_once()
    finally:
        stream.close()
