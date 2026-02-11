from PyQt6.QtCore import QObject

class BaseWorker(QObject):
    """
    A base worker class that inherits from QObject to leverage Qt's signals and slots mechanism.
    """
    def __init__(self, *args, **kwargs):
        """
        Initializer that sets up instance state.
        """
        super().__init__(*args, **kwargs)
        self._cancel_requested = False

    def request_cancel(self):
        self._cancel_requested = True

    def is_cancel_requested(self):
        return self._cancel_requested

    def run(self):
        """
        The main work method.
        """
        raise NotImplementedError
