import json
import time
from enum import Enum, auto

from mobilepipe.comm.comm_utils import CommHandler


class TimeKeeper:
    """
    Handles timekeeping for experiments; measures idle time, batch times, comms delay, etc.

    Times are stored as pairs: [start, end]

    During Dynamic Pipeline Configurations:

    - track initial model weights offload time
    - track intermediate configuration sync times

    - track host time per s1_forward microbatch
    - track host idle time waiting for s2_backward intermediate grads
    - track host time per s1_backward microbatch
    - track host time full batch
    * receive ios times for each integrated s2_forward_backward microbatch
    - track host time for optimization step
    * receive ios time for optimization step
    """

    class CATS(Enum):
        INIT_MODEL_OFFLOAD = auto()
        S1_FORWARD_MICROBATCH = auto()
        S1_BACKWARD_MICROBATCH = auto()
        WAITING_FOR_GRAD = auto()
        OPTIMIZATION_STEP = auto()
        INTERMEDIATE_CONFIG_SYNC = auto()
        FULL_BATCH = auto()
        TRAINING_RUN = auto()

    class TimePair:
        def __init__(self, start=-1.0, end=-1.0):
            self.time = [start, end]

        def set_start(self, start):
            self.time[0] = start

        def get_start(self):
            return self.time[0]

        def set_end(self, end):
            self.time[1] = end

        def get_end(self):
            return self.time[1]

        def get_time(self):
            return self.time

    def __init__(self):
        self.times = {cat: [] for cat in self.CATS}
        self.received_ios_times = None

    def start(self, cat: CATS):
        self.times[cat].append(self.TimePair(start=self._cur_time()))

    def end(self, cat: CATS):
        self.times[cat][-1].set_end(end=self._cur_time())

    @staticmethod
    def _cur_time():
        return time.time()

    def receive_ios_times(self, comm_handler: CommHandler):
        """
        Receive time data sent by the iOS side.
        """
        total_values = int(comm_handler.receive_double())

        time_values = []
        for _ in range(total_values):
            time_values.append(comm_handler.receive_double())

        received_times = {
            'IOS_WAITING_FOR_INPUT': [],
            'IOS_S2_COMBINED_MICROBATCH': [],
            'IOS_OPTIMIZATION_STEP': []
        }

        idx = 0
        nof = (len(time_values) - 2) // 2
        while idx + 1 < len(time_values):
            # Check if only 2 values remain, which would be OPTIMIZATION_STEP
            if len(time_values) - idx == 2:
                break
            if idx < nof:
                start = time_values[idx]
                end = time_values[idx + 1]
                received_times['IOS_WAITING_FOR_INPUT'].append([start, end])
                idx += 2
            else:
                start = time_values[idx]
                end = time_values[idx + 1]
                received_times['IOS_S2_COMBINED_MICROBATCH'].append([start, end])
                idx += 2
        if idx + 1 < len(time_values):
            start = time_values[idx]
            end = time_values[idx + 1]
            received_times['IOS_OPTIMIZATION_STEP'].append([start, end])
            idx += 2

        self.received_ios_times = received_times

    def clear(self):
        self.times = {cat: [] for cat in self.CATS}
        self.received_ios_times = None

    def write_to_file(self, path: str):
        all_times = {}
        if self.received_ios_times:
            all_times.update(self.received_ios_times)
        for cat, time_pairs in self.times.items():
            cat_name = cat.name
            time_data = [pair.get_time() for pair in time_pairs]
            all_times[f"HOST_{cat_name}"] = time_data
        with open(path, 'w') as f:
            json.dump(all_times, f)

