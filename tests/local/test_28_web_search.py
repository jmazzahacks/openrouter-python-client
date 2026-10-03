"""Search option serialization, citation preservation, and high-level state."""

import json
from copy import deepcopy
from unittest.mock import Mock, patch

import pytest

from openrouter_client import (
    Annotation,
    ToolLoop,
    UrlCitation,
    UrlCitationAnnotation,
    WebSearchOptions,
    WebSearchPlugin,
    get_model,
)
from openrouter_client.auth import AuthManager
from openrouter_client.endpoints.chat import ChatEndpoint
from openrouter_client.exceptions import APIError
from openrouter_client.http import HTTPManager
from openrouter_client.models import ChatCompletionRequest
from openrouter_client.models.core import Message

CITATION = {
    "type": "url_citation",
    "url_citation": {
        "url": "https://example.com/news",
        "title": "News report",
        "content": "An excerpt from the source.",
        "start_index": 0,
        "end_index": 6,
        "future_detail": "retained",
    },
}
UNKNOWN = {"type": "future_annotation", "payload": {"nested": [1, 2]}}
OPTIONS = {
    "id": "web",
    "engine": "exa",
    "mode": "auto",
    "max_results": 5,
    "search_prompt": "Use these sources:",
    "include_domains": ["example.com"],
    "exclude_domains": ["ads.example.com"],
}
SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}}


def body(content="Answer", annotations=None, cost=0.008):
    message = {"role": "assistant", "content": content}
    if annotations is not None:
        message["annotations"] = deepcopy(annotations)
    return {
        "id": "chat-search",
        "created": 1,
        "model": "test/model",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
            "cost": cost,
        },
    }


def make_endpoint(*bodies):
    http = Mock(spec=HTTPManager)
    http.base_url = "https://openrouter.ai/api/v1"
    http.post.side_effect = [Mock(json=Mock(return_value=b)) for b in bodies]
    endpoint = ChatEndpoint(AuthManager(api_key="test-key"), http)
    return endpoint, http


def make_wrapper(conversation, *bodies):
    endpoint, http = make_endpoint(*bodies)
    client = Mock()
    client.chat = endpoint
    model = get_model("test/model", client)
    return (model.conversation() if conversation else model), http


def assert_citation(annotation):
    assert isinstance(annotation, UrlCitationAnnotation)
    assert annotation.url_citation.url == CITATION["url_citation"]["url"]
    assert annotation.model_dump(exclude_none=True) == CITATION


@pytest.mark.parametrize("typed", [False, True])
@pytest.mark.parametrize("validate", [False, True])
def test_search_options_reach_wire_and_citations_are_typed(typed, validate):
    endpoint, http = make_endpoint(body(annotations=[CITATION, UNKNOWN]))
    plugin = WebSearchPlugin(**OPTIONS) if typed else OPTIONS
    options = (
        WebSearchOptions(search_context_size="high")
        if typed
        else {"search_context_size": "high"}
    )
    result = endpoint.create(
        model="test/model:online",
        messages=[{"role": "user", "content": "news"}],
        plugins=[plugin, {"id": "response-healing", "future_option": True}],
        web_search_options=options,
        validate_request=validate,
    )
    sent = http.post.call_args.kwargs["json"]
    assert sent["model"] == "test/model:online"
    assert sent["plugins"] == [
        OPTIONS,
        {"id": "response-healing", "future_option": True},
    ]
    assert sent["web_search_options"] == {"search_context_size": "high"}
    assert_citation(result.choices[0].message.annotations[0])
    unknown = result.choices[0].message.annotations[1]
    assert isinstance(unknown, Annotation)
    assert unknown.model_dump() == UNKNOWN
    assert result.usage.cost == 0.008


def test_request_model_round_trip_preserves_options_and_unknown_plugins():
    raw = {
        "model": "test/model",
        "messages": [{"role": "user", "content": "news"}],
        "plugins": [OPTIONS, {"id": "new-plugin", "setting": 1}],
        "web_search_options": {"search_context_size": "low", "future_option": True},
    }
    result = ChatCompletionRequest.model_validate(raw).model_dump(exclude_none=True)
    assert result == raw


def test_typed_search_options_preserve_unknown_null_settings_on_wire():
    endpoint, http = make_endpoint(body())
    endpoint.create(
        model="test/model",
        messages=[],
        plugins=[WebSearchPlugin(engine="exa", future_setting=None)],
        web_search_options=WebSearchOptions(future_setting=None),
    )
    sent = http.post.call_args.kwargs["json"]
    assert sent["plugins"] == [{"id": "web", "engine": "exa", "future_setting": None}]
    assert sent["web_search_options"] == {"future_setting": None}


@pytest.mark.parametrize("annotations", [None, []])
def test_message_without_citations(annotations):
    message = Message(role="assistant", content="Answer", annotations=annotations)
    assert message.annotations == annotations


def test_minimal_citation_and_unknown_fields_survive_message_round_trip():
    raw = {"type": "url_citation", "url_citation": {"url": "https://example.com"}}
    message = Message(role="assistant", annotations=[raw, UNKNOWN])
    assert isinstance(message.annotations[0], UrlCitationAnnotation)
    assert message.annotations[0].url_citation.content is None
    assert message.model_dump(exclude_none=True)["annotations"] == [raw, UNKNOWN]


@pytest.mark.parametrize("stream", [False, True])
def test_typed_annotations_in_message_dicts_are_json_serializable(stream):
    endpoint, http = make_endpoint(body())
    annotation = UrlCitationAnnotation(
        url_citation=UrlCitation(url="https://example.com", content=None)
    )
    with patch(
        "openrouter_client.endpoints.chat.StreamingChatCompletionsRequest"
    ) as streamer:
        streamer.return_value.stream.return_value = iter([])
        endpoint.create(
            model="test/model",
            messages=[
                {"role": "assistant", "content": "news", "annotations": [annotation]}
            ],
            stream=stream,
        )
        sent = (
            streamer.call_args.kwargs["messages"]
            if stream
            else http.post.call_args.kwargs["json"]["messages"]
        )
    assert json.loads(json.dumps(sent))[0]["annotations"] == [
        {
            "type": "url_citation",
            "url_citation": {"url": "https://example.com", "content": None},
        }
    ]


@pytest.mark.parametrize("tool_loop", [False, True])
@pytest.mark.parametrize("schema", [None, SCHEMA])
def test_conversation_replays_explicit_null_annotation_fields(tool_loop, schema):
    annotations = [
        {"type": "future_annotation", "payload": None},
        {
            "type": "url_citation",
            "url_citation": {
                "url": "https://example.com",
                "content": None,
                "future_detail": None,
            },
            "future_field": None,
        },
    ]
    content = '{"answer": "news"}' if schema else "news"
    responses = [body(content, annotations)]
    if tool_loop and schema:
        responses.append(body(content, annotations))
    responses.append(body("follow up"))
    conversation, http = make_wrapper(True, *responses)
    loop = ToolLoop(tools=[], handlers={}) if tool_loop else None
    conversation.prompt("news", schema=schema, tool_loop=loop)
    assert conversation.messages[-1]["annotations"] == annotations
    conversation.prompt("follow up")
    sent = http.post.call_args.kwargs["json"]["messages"]
    cited_messages = [m for m in sent if "annotations" in m]
    assert len(cited_messages) == (2 if tool_loop and schema else 1)
    assert all(m["annotations"] == annotations for m in cited_messages)


def test_stream_options_and_annotations_survive_endpoint_normalization():
    endpoint, _ = make_endpoint()
    chunk = {
        "id": "search-stream",
        "created": 1,
        "model": "test/model",
        "choices": [{"index": 0, "delta": {"annotations": [CITATION, UNKNOWN]}}],
    }
    with patch(
        "openrouter_client.endpoints.chat.StreamingChatCompletionsRequest"
    ) as streamer:
        streamer.return_value.stream.return_value = iter([chunk])
        chunks = list(
            endpoint.create(
                model="test/model",
                messages=[],
                stream=True,
                plugins=[WebSearchPlugin(**OPTIONS)],
                web_search_options=WebSearchOptions(search_context_size="low"),
            )
        )
    assert streamer.call_args.kwargs["params"]["plugins"] == [OPTIONS]
    assert streamer.call_args.kwargs["params"]["web_search_options"] == {
        "search_context_size": "low"
    }
    assert_citation(chunks[0].choices[0].delta.annotations[0])
    assert chunks[0].choices[0].delta.annotations[1].model_dump() == UNKNOWN


@pytest.mark.parametrize("conversation", [False, True])
@pytest.mark.parametrize("schema", [None, SCHEMA])
def test_high_level_options_annotations_cost_and_return_types(conversation, schema):
    content = '{"answer": "news"}' if schema else "news"
    wrapper, http = make_wrapper(
        conversation, body(content, [CITATION, UNKNOWN]), body("plain")
    )
    assert wrapper.last_annotations == []
    result = wrapper.prompt(
        "news",
        schema=schema,
        plugins=[WebSearchPlugin(**OPTIONS)],
        web_search_options=WebSearchOptions(search_context_size="medium"),
    )
    assert result == ({"answer": "news"} if schema else "news")
    assert_citation(wrapper.last_annotations[0])
    assert wrapper.last_annotations[1].model_dump() == UNKNOWN
    assert wrapper.last_usage.cost == 0.008
    assert http.post.call_args.kwargs["json"]["plugins"] == [OPTIONS]
    if conversation:
        assert wrapper.messages[-1]["annotations"] == [CITATION, UNKNOWN]
    assert wrapper.prompt("follow up") == "plain"
    assert wrapper.last_annotations == []
    if conversation:
        assert wrapper.total_cost == 0.016
        assert http.post.call_args.kwargs["json"]["messages"][1]["annotations"] == [
            CITATION,
            UNKNOWN,
        ]


@pytest.mark.parametrize("conversation", [False, True])
@pytest.mark.parametrize("tool_loop", [False, True])
def test_failed_prompt_keeps_previous_annotations_and_usage(conversation, tool_loop):
    wrapper, _ = make_wrapper(
        conversation, body(annotations=[CITATION]), body("not JSON"), body("not JSON")
    )
    loop = ToolLoop(tools=[], handlers={}) if tool_loop else None
    wrapper.prompt("news", tool_loop=loop)
    previous_usage = wrapper.last_usage
    previous_annotations = list(wrapper.last_annotations)
    with pytest.raises(APIError):
        wrapper.prompt("structured", schema=SCHEMA, tool_loop=loop)
    assert wrapper.last_usage == previous_usage
    assert wrapper.last_annotations == previous_annotations


@pytest.mark.parametrize("conversation", [False, True])
@pytest.mark.parametrize("schema", [None, SCHEMA])
def test_tool_loop_exposes_final_message_citations(conversation, schema):
    first = body(None, [UNKNOWN])
    first["choices"][0]["message"]["tool_calls"] = [
        {
            "id": "call-1",
            "type": "function",
            "function": {"name": "lookup", "arguments": "{}"},
        }
    ]
    responses = [first, body("news", [CITATION])]
    if schema:
        responses.append(body('{"answer": "news"}', [CITATION, UNKNOWN]))
    wrapper, http = make_wrapper(conversation, *responses)
    loop = ToolLoop(
        tools=[
            {
                "type": "function",
                "function": {"name": "lookup", "parameters": {"type": "object"}},
            }
        ],
        handlers={"lookup": lambda: "facts"},
    )
    wrapper.prompt(
        "news", tool_loop=loop, schema=schema, plugins=[WebSearchPlugin(**OPTIONS)]
    )
    assert_citation(wrapper.last_annotations[0])
    assert len(wrapper.last_annotations) == (2 if schema else 1)
    assert wrapper.last_usage.cost == pytest.approx(0.008 * len(responses))
    assert all(
        call.kwargs["json"]["plugins"] == [OPTIONS] for call in http.post.call_args_list
    )
    if conversation:
        assert wrapper.messages[1]["annotations"] == [UNKNOWN]


@pytest.mark.parametrize("use_loop", [False, True])
def test_server_tool_dict_passes_through_without_client_handler(use_loop):
    endpoint, http = make_endpoint(body(annotations=[CITATION]))
    tool = {
        "type": "openrouter:web_search",
        "parameters": {"engine": "exa", "max_results": 5},
    }
    if use_loop:
        client = Mock()
        client.chat = endpoint
        model = get_model("test/model", client)
        assert (
            model.prompt("news", tool_loop=ToolLoop(tools=[tool], handlers={}))
            == "Answer"
        )
        annotations = model.last_annotations
    else:
        result = endpoint.create(model="test/model", messages=[], tools=[tool])
        annotations = result.choices[0].message.annotations
    assert http.post.call_args.kwargs["json"]["tools"] == [tool]
    assert_citation(annotations[0])
