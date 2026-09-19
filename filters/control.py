"""ControlMixin: cheap per-frame polling of a flat control.json file for live
filter reconfiguration, without restarting filter processes. Caches by file
mtime so it's not a disk read every frame.
"""

import json
import os


class ControlMixin:
    _control_path = "control.json"
    _control_mtime = 0
    _control = {}

    def get_control(self):
        try:
            mtime = os.path.getmtime(self._control_path)
            if mtime != self._control_mtime:
                with open(self._control_path) as f:
                    self._control = json.load(f)
                self._control_mtime = mtime
        except FileNotFoundError:
            pass
        return self._control
