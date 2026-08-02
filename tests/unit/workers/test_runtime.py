"""Tests that the JSONL transport is independent of model endpoints."""

from io import StringIO
import json
import unittest

from grounding_control.transport import (
    DEFAULT_RESPONSE_PREFIX,
    WorkerRequestError,
    process_request_line,
    serve_jsonl,
)


class _Engine:
    def handle(self, payload):
        operation = payload.get('operation')
        if operation == 'echo':
            return {'value': payload.get('value')}
        if operation == 'shutdown':
            return {'shutdown': True}
        raise WorkerRequestError('unsupported')


class WorkerRuntimeTests(unittest.TestCase):
    def test_transport_wraps_any_engine(self):
        response = process_request_line(
            json.dumps({
                'request_id': 'one',
                'operation': 'echo',
                'value': 7,
            }),
            _Engine(),
            StringIO(),
        )
        self.assertTrue(response['ok'])
        self.assertEqual(response['request_id'], 'one')
        self.assertEqual(response['value'], 7)

    def test_server_lifecycle_does_not_depend_on_model_roles(self):
        output = StringIO()
        serve_jsonl(
            _Engine(),
            stdin=[
                '{"operation":"echo","value":1}\n',
                '{"operation":"shutdown"}\n',
                '{"operation":"echo","value":2}\n',
            ],
            stdout=output,
            stderr=StringIO(),
        )
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith(DEFAULT_RESPONSE_PREFIX))


if __name__ == '__main__':
    unittest.main()
