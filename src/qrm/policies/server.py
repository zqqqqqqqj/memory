"""Single synchronous rollout client; actor weights remain fixed while collecting."""
from contextlib import contextmanager
import logging
import threading
import traceback
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve
from openpi_client import msgpack_numpy


@contextmanager
def policy_server(runtime, host="127.0.0.1", port=8011):
    lock = threading.Lock()

    def handler(connection):
        if not lock.acquire(blocking=False):
            connection.close(code=1013, reason="One rollout client is supported")
            return
        try:
            connection.send(msgpack_numpy.packb({"protocol": "qrm-v1"}))
            for message in connection:
                try:
                    output = runtime.request(msgpack_numpy.unpackb(message))
                    connection.send(msgpack_numpy.packb(output))
                except Exception:
                    connection.send(traceback.format_exc())
                    break
        except ConnectionClosed:
            if runtime.active:
                logging.getLogger(__name__).warning('Policy client disconnected before episode finish')
        finally:
            lock.release()

    with serve(handler, host, port, max_size=None) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server.socket.getsockname()[1]
        finally:
            server.shutdown()
            thread.join(timeout=10)
