import json

import pytest

from openrouter_client.models.llm import (
    build_json_schema_response_format,
    parse_schema_response,
)
from openrouter_client.exceptions import APIError


SCHEMA = {"type": "object", "properties": {"name": {"type": "string"}}}


class Test_BuildJsonSchemaResponseFormat_01_NominalBehaviors:
    """Tests for build_json_schema_response_format."""

    def test_default_name(self):
        result = build_json_schema_response_format(SCHEMA)
        assert result == {
            "type": "json_schema",
            "json_schema": {"name": "response_schema", "schema": SCHEMA},
        }

    def test_custom_name(self):
        result = build_json_schema_response_format(SCHEMA, name="person")
        assert result["json_schema"]["name"] == "person"
        assert result["json_schema"]["schema"] is SCHEMA


class Test_ParseSchemaResponse_01_NominalBehaviors:
    """Valid inputs return a dict."""

    def test_dict_passthrough(self):
        payload = {"name": "Ada"}
        assert parse_schema_response(payload, SCHEMA) == payload

    def test_valid_json_string(self):
        assert parse_schema_response('{"name": "Ada"}', SCHEMA) == {"name": "Ada"}


class Test_ParseSchemaResponse_02_ErrorHandling:
    """Invalid inputs raise APIError."""

    @pytest.mark.parametrize("bad", ["", "   "])
    def test_empty_response_raises(self, bad):
        with pytest.raises(APIError):
            parse_schema_response(bad, SCHEMA)

    def test_invalid_json_raises(self):
        with pytest.raises(APIError):
            parse_schema_response("{not json", SCHEMA)

    def test_non_object_json_raises(self):
        with pytest.raises(APIError):
            parse_schema_response("[1, 2, 3]", SCHEMA)

    def test_unexpected_type_raises(self):
        with pytest.raises(APIError):
            parse_schema_response(123, SCHEMA)
