import os
import copy
import yaml  
import json  
import time  
import random
import string 

from dataclasses import dataclass


class Generator:
    """
    Generator class responsible for generating server definitions, process patterns, 
    and populating processes based on a configuration.
    """

    @dataclass
    class EventDTO:
        """
        Data Transfer Object (DTO) for events in the system.
        """
        source: str  # Source server
        target: str  # Target server
        time: int  # Timestamp of the event
        type: str  # Type of the event ('Request' or 'Response')
        id: int  # Identifier for the event


    def __init__(self):
        """
        Initializes the Generator with character set for random string generation.
        """
        self.chars = string.ascii_uppercase + string.digits  # Uppercase letters and digits


    def _set_(self, config_path="config.yaml"):
        """
        Sets the configuration for the generator by reading a YAML config file.
        
        Args:
            config_path (str): Path to the configuration file.
        """
        # Load configuration from YAML file
        with open(config_path, 'r') as file:
            self.CONFIG = yaml.safe_load(file)

        # Load request time ranges and weights from the config
        self.REQUEST_TIME_RANGES = [
            self.CONFIG["REQUEST_TIME_RANGES"][key] for key in self.CONFIG["REQUEST_TIME_RANGES"].keys()
        ]
        self.REQUEST_TIME_WEIGHTS = [
            self.CONFIG["REQUEST_TIME_WEIGHTS"][key] for key in self.CONFIG["REQUEST_TIME_WEIGHTS"].keys()
        ]

        # Define servers with random prefixes and suffixes
        self.SERVER_DEFINITIONS = {
            f"{self.random_string()}": {
                "prefixes": [
                    self.random_string() for _ in range(
                        random.randrange(*self.CONFIG["TYPE_SUFFIXES_PREFIXES_NUMBER_RANGE"])
                    )
                ],
                "suffixes": [
                    self.random_string() for _ in range(
                        random.randrange(*self.CONFIG["TYPE_SUFFIXES_PREFIXES_NUMBER_RANGE"])
                    )
                ],
                "instances": set()
            }
            for _ in range(self.CONFIG["TYPE_NUMBER"])
        }

        # Add server instances to the definitions
        for server_type, additions in self.SERVER_DEFINITIONS.items():
            self.add_server(server_type, server_type)

            if self.CONFIG["TYPE_PREFIX_ENABLED"]:
                for prefix in additions['prefixes']:
                    self.add_server(server_type, prefix + server_type)

                    if self.CONFIG["TYPE_SUFFIX_ENABLED"]:
                        for suffix in additions['suffixes']:
                            self.add_server(server_type, prefix + server_type + suffix)

            if self.CONFIG["TYPE_SUFFIX_ENABLED"]:
                for suffix in additions['suffixes']:
                    self.add_server(server_type, server_type + suffix)

        # Save server definitions to a JSON file
        types_path = "./output/types.json"
        os.makedirs(os.path.dirname(types_path), exist_ok=True)
        with open(types_path, "w") as f:
            json.dump(self.SERVER_DEFINITIONS, f, default=self._serialize_set_, indent=4)

        # Initialize patterns list
        self.patterns = []


    def _serialize_set_(self, obj):
        """
        Helper method to serialize sets as lists when saving to JSON.

        Args:
            obj: Object to serialize.

        Returns:
            Serialized object.
        """
        if isinstance(obj, set):
            return list(obj)
        return obj


    def random_string(self):
        """
        Generates a random string based on the character set and length range in the config.

        Returns:
            str: Randomly generated string.
        """
        return ''.join(random.choices(
            self.chars,
            k=random.randrange(*self.CONFIG["TYPE_STRING_LENGTH_RANGE"])
        ))


    def add_server(self, server_type, server_name):
        """
        Adds a server to the server definitions.

        Args:
            server_type (str): Type of the server.
            server_name (str): Name of the server.
        """
        # Add server to instances if not skipping
        if len(self.SERVER_DEFINITIONS[server_type]["instances"]) == 0 or\
                random.random() > self.CONFIG["TYPE_SKIP_CHANCE"] / 100:
            self.SERVER_DEFINITIONS[server_type]["instances"].add(server_name)

    def add_process_event(self, source, target, id, depth, length, process, servers):
        """
        Recursively adds process events to a process.

        Args:
            source (str): Source server.
            target (str): Target server.
            id (int): Event ID.
            depth (int): Current depth of the process.
            length (int): Current length of the process.
            process (list): List of events in the process.
            servers (set): Set of available servers.

        Returns:
            int: Updated length of the process.
        """
        # Remove the target from the set of servers
        servers = set(servers) - set([target])
        # Add a request event
        process.append(
            self.EventDTO(
                source=source,
                target=target,
                time=-1,
                type='Request',
                id=id
            )
        )

        # Recursively add more events based on config constraints
        while True:
            if length >= self.CONFIG["PROCESS_MAX_LENGTH"] or\
                    depth >= self.CONFIG["PROCESS_MAX_DEPTH"] or\
                    random.random() < self.CONFIG["REQUEST_END_CHANCE"] / 100:
                # Add a response event to end the process
                process.append(
                    self.EventDTO(
                        source=target,
                        target=source,
                        time=-1,
                        type='Response',
                        id=id
                    )
                )
                return length
            else:
                # Recursively add a process event
                length = self.add_process_event(
                    target,
                    random.choice(tuple(servers)),
                    id,
                    depth + 1,
                    length + 2,
                    process,
                    servers
                )

    def random_milliseconds(self, start=False):
        """
        Generates a random number of milliseconds based on the request time ranges and weights.

        Args:
            start (bool): If True, return a fixed start time. Defaults to False.

        Returns:
            int: Random number of milliseconds.
        """
        if start:
            return 1000
        else:
            return random.randrange(
                *random.choices(
                    self.REQUEST_TIME_RANGES,
                    weights=self.REQUEST_TIME_WEIGHTS
                )[0]
            )

    def generate_process_patterns(self, pattern_number=None, save=True):
        """
        Generates process patterns and optionally saves them to a file.

        Args:
            pattern_number (int): Number of patterns to generate. Defaults to config value.
            save (bool): If True, save the patterns to a file. Defaults to True.
        """
        if not pattern_number:
            pattern_number = self.CONFIG["PROCESS_PATTERN_NUMBER"]

        id = -1

        # Generate the process patterns
        for _ in range(pattern_number):
            id += 1
            process = []
            self.add_process_event(
                None,
                random.choice(tuple(self.SERVER_DEFINITIONS.keys())),
                id,
                0,
                2,
                process,
                self.SERVER_DEFINITIONS.keys()
            )
            self.patterns.append(process)

        unique_patterns = set()
        patterns = list()
        id = 0

        # Filter unique patterns and reassign IDs
        for pattern in self.patterns:
            unique_patterns_length = len(unique_patterns)
            order = [x.target for x in pattern if x.type == 'Request']
            unique_patterns.add(str(order))
            if len(unique_patterns) != unique_patterns_length:
                for event in pattern:
                    event.id = id
                id += 1
                patterns.append(pattern)

        self.patterns = patterns

        if save:
            # Save patterns to a CSV file
            file_path = "./output/patterns.csv"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as f:
                for pattern in self.patterns:
                    for event in pattern:
                        f.write(
                            f"<{event.source},{event.target},{event.time},{event.type},{event.id}>\n"
                        )

    def populate_processes(self, process_number=None, save=True):
        """
        Populates processes based on the generated patterns and optionally saves them to files.

        Args:
            process_number (int): Number of processes to generate. Defaults to config value.
            save (bool): If True, save the processes to files. Defaults to True.
        """
        if not process_number:
            process_number = self.CONFIG["PROCESSES_TO_GENERATE"]

        log = []
        start_time = round(time.time() * 1000)
        id = 0

        if save:
            # Initialize stats CSV file
            stats_path = "./output/stats.csv"
            os.makedirs(os.path.dirname(stats_path), exist_ok=True)
            with open(stats_path, "w") as f:
                f.write("process_id,pattern_id,duration,call_order\n")

        for _ in range(process_number):
            pattern = random.choice(self.patterns)
            pattern_copy = copy.deepcopy(pattern)
            event_time = start_time = start_time + self.random_milliseconds(start=True)

            for i, event in enumerate(pattern_copy):
                event.time = event_time
                event.id = id

                if event.type == 'Request':
                    resp = [x for x in pattern_copy[i:] if x.source == event.target and x.type == "Response"][0]
                    event.target = random.choice(tuple(self.SERVER_DEFINITIONS[event.target]["instances"]))
                    resp.source = event.target
                    resp.target = event.source

                if event.target:
                    next_req = pattern_copy[i + 1]
                    next_req.source = event.target

                duration = event_time - start_time
                event_time += self.random_milliseconds()

            log.extend(pattern_copy)

            if save:
                with open(stats_path, "a") as f:
                    f.write(f"{id},{pattern[0].id},{duration},{[x.target for x in pattern_copy if x.type == 'Request']}\n")
            id += 1

        if save:
            # Save unsorted log to a CSV file
            file_path = "./output/unsorted_log.csv"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as f:
                for event in log:
                    f.write(f"<{event.source},{event.target},{event.time},{event.type},{event.id}>\n")

        sorted_log = sorted(log, key=lambda x: x.time)

        if save:
            # Save sorted log to a CSV file
            file_path = "./output/log.csv"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as f:
                for event in sorted_log:
                    f.write(f"<{event.source},{event.target},{event.time},{event.type},{event.id}>\n")

        return sorted_log


def main():
    """
    Main function to initialize the generator and generate process patterns and processes.
    """
    st = time.time()
    generator = Generator()
    generator._set_()
    generator.generate_process_patterns()
    generator.populate_processes()
    print("--- %s seconds ---" % (time.time() - st))


if __name__ == "__main__":
    main()
