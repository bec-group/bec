from __future__ import annotations

import abc
import sys
import threading


class MessageObject:
    def __init__(self, value, topic) -> None:
        self.topic = topic
        self._value = value

    @property
    def value(self):
        return self._value


class ConnectorBase(abc.ABC):
    """
    ConnectorBase implements producer and consumer clients for communicating with a broker.
    One ought to inherit from this base class and provide at least customized producer and consumer methods.

    """

    def __init__(self, bootstrap_server: list):
        self.bootstrap = bootstrap_server
        self._threads = []

    def producer(self, **kwargs) -> ProducerConnector:
        raise NotImplementedError

    def consumer(self, **kwargs) -> ConsumerConnectorThreaded:
        raise NotImplementedError

    def shutdown(self):
        for t in self._threads:
            t.signal_event.set()
            t.join()

    def raise_warning(self, msg):
        raise NotImplementedError

    def send_log(self, msg):
        raise NotImplementedError

    def raise_error(self, msg):
        raise NotImplementedError


class ProducerConnector(abc.ABC):
    def send(self, topic: str, msg) -> None:
        raise NotImplemented


class ConsumerConnector(abc.ABC):
    def __init__(
        self,
        bootstrap_server,
        topics=None,
        pattern=None,
        group_id=None,
        event=None,
        cb=None,
        **kwargs,
    ):
        """
        ConsumerConnector class defines the communication with the broker for consuming messages.
        An implementation ought to inherit from this class and implement the initialize_connector and poll_messages methods.

        Args:
            bootstrap_server: list of bootstrap servers, e.g. ["localhost:9092", "localhost:9093"]
            topics: the topic(s) to which the connector should attach
            event: external event to trigger start and stop of the connector
            cb: callback function; will be triggered from within poll_messages
            kwargs: additional keyword arguments

        """
        super().__init__()
        self.bootstrap = bootstrap_server
        self.topics = topics
        self.pattern = pattern
        self.group_id = group_id
        self.connector = None
        self.cb = cb
        self.kwargs = kwargs

    def initialize_connector(self) -> None:
        """
        initialize the connector instance self.connector
        The connector will be initialized once the thread is started
        """
        raise NotImplementedError

    def poll_messages(self) -> None:
        """
        Poll messages from self.connector and call the callback function self.cb

        """
        messages = self.connector.poll(10.0)
        for key, values in messages.items():
            for msg in values:
                self.cb(msg, **self.kwargs)


class ConsumerConnectorThreaded(threading.Thread, abc.ABC):
    def __init__(
        self,
        bootstrap_server,
        topics=None,
        pattern=None,
        group_id=None,
        event=None,
        cb=None,
        **kwargs,
    ):
        """
        ConsumerConnector class defines the threaded communication with the broker for consuming messages.
        An implementation ought to inherit from this class and implement the initialize_connector and poll_messages methods.
        Once started, the connector is expected to poll new messages until the signal_event is set.

        Args:
            bootstrap_server: list of bootstrap servers, e.g. ["localhost:9092", "localhost:9093"]
            topics: the topic(s) to which the connector should attach
            event: external event to trigger start and stop of the connector
            cb: callback function; will be triggered from within poll_messages
            kwargs: additional keyword arguments

        """
        super().__init__(daemon=True)
        self.bootstrap = bootstrap_server
        self.topics = topics
        self.pattern = pattern
        self.group_id = group_id
        self.signal_event = event if event is not None else threading.Event()
        self.connector = None
        self.cb = cb
        self.kwargs = kwargs

    def run(self):
        self.initialize_connector()

        while True:
            try:
                self.poll_messages()
            except KeyError:
                print("Exception", sys.exc_info())
            finally:
                if self.signal_event.is_set():
                    self.shutdown()
                    break

    def initialize_connector(self) -> None:
        """
        initialize the connector instance self.connector
        The connector will be initialized once the thread is started
        """
        raise NotImplementedError

    def poll_messages(self) -> None:
        """
        Poll messages from self.connector and call the callback function self.cb

        """
        messages = self.connector.poll(10.0)
        for key, values in messages.items():
            for msg in values:
                self.cb(msg, **self.kwargs)

    def shutdown(self):
        self.connector.close()

    # def stop(self) -> None:
    #     """
    #     Stop consumer
    #     Returns:
    #
    #     """
    #     self.signal_event.set()
    #     self.connector.close()
    #     self.join()