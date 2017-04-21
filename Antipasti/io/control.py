import threading
import yaml
import os
import time

from .. import backend as A


class SwitchBoard(object):
    """Streams variable values from a YAML file."""
    # TODO Documentation
    def __init__(self, switches, yaml_file, session=None):
        # Private
        self._file_last_modified = 0
        self._observer_thread = None
        self._stop_observing = threading.Event()
        self._latency = 0.3
        # Public
        self.switches = switches
        self.yaml_file = yaml_file
        self.session = session if session is not None else A.Session.session

    def add_switch(self, switch_name, switch_variable=None, **switch_variable_init_kwargs):
        # Make switch variable if not provided
        if switch_variable is None:
            switch_variable = A.variable(**switch_variable_init_kwargs)
        # Add variable to switches
        self.switches.update({switch_name: switch_variable})
        # Return the variable made
        return switch_variable

    @property
    def file_has_changed(self):
        file_last_modified_according_to_os = os.stat(self.yaml_file).st_mtime
        if file_last_modified_according_to_os != self._file_last_modified:
            self._file_last_modified = file_last_modified_according_to_os
            return True
        else:
            return False

    @A.with_master_graph
    def _observe(self):
        # We don't use self._stop_observing.wait(timeout=self._latency) because the Python 2.X
        # implementation keeps reacquiring GIL.
        while True:
            # Break if required
            if self._stop_observing.is_set():
                break
            # Be kind
            time.sleep(self._latency)
            # Break if required
            if self._stop_observing.is_set():
                break
            # Check if file has changed
            if self.file_has_changed:
                # Read file
                with open(self.yaml_file, 'r') as f:
                    updates = yaml.load(f)
                # Update switches
                for switch_name, switch_variable in self.switches.items():
                    switch_update = updates.get(switch_name)
                    if switch_update is not None:
                        A.set_value(switch_variable, switch_update, session=self.session)
            # Break if required
            if self._stop_observing.is_set():
                break

    @property
    def observing(self):
        return self._observer_thread is not None and self._observer_thread.is_alive()

    def start_observer(self):
        self._observer_thread = threading.Thread(target=self._observe)
        self._observer_thread.start()

    def stop_observer(self):
        if self.observing:
            # Set stop obs signal
            self._stop_observing.set()
            # Wait for thread to join
            self._observer_thread.join()
