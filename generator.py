import os
import copy
import yaml
import json
import time
import random
import string

from dataclasses import dataclass


class Generator():

    @dataclass
    class EventDto:
        source: str
        target: str
        time: int
        type: str
        id: int

    
    def __init__(self):
        self.chars = string.ascii_uppercase + string.digits

    
    def _set_(self, config_path="config.yaml"):

        with open(config_path, 'r') as file:
            self.CONFIG = yaml.safe_load(file)

        self.REQUEST_TIME_RANGES = [
            self.CONFIG["REQUEST_TIME_RANGES"][key] for key in self.CONFIG["REQUEST_TIME_RANGES"].keys()
        ]
        self.REQUEST_TIME_WEIGHTS = [
            self.CONFIG["REQUEST_TIME_WEIGHTS"][key] for key in self.CONFIG["REQUEST_TIME_WEIGHTS"].keys()
        ]

        self.SERVER_DEFINITIONS = {
            f"{self.random_string()}": {
                "prefixes" : [
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

        for server_type, additions in self.SERVER_DEFINITIONS.items():
            self.add_server(server_type, server_type)

            for prefix in additions['prefixes']:
                self.add_server(server_type, prefix + server_type)

                for suffix in additions['suffixes']:
                    self.add_server(server_type, prefix + server_type + suffix)

            for suffix in additions['suffixes']:
                self.add_server(server_type, server_type + suffix)

        types_path = "./output/types.json"
        os.makedirs(os.path.dirname(types_path), exist_ok=True)
        with open(types_path, "w") as f: 
            json.dump(self.SERVER_DEFINITIONS, f, default=self._serialize_set_, indent=4)

        self.patterns = []


    def _serialize_set_(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return obj


    def random_string(self):
        return ''.join(random.choices(
            self.chars, 
            k=random.randrange(*self.CONFIG["TYPE_STRING_LENGTH_RANGE"])
        ))


    def add_server(self, server_type, server_name):

        if len(self.SERVER_DEFINITIONS[server_type]["instances"]) == 0 or\
              random.random() > self.CONFIG["TYPE_SKIP_CHANCE"] / 100:
            self.SERVER_DEFINITIONS[server_type]["instances"].add(server_name)


    def add_process_event(self, source, target, id, depth, length, process, servers):
        servers = set(servers) - set([target])
        process.append(
            self.EventDto(
                source=source,
                target=target,
                time=-1,
                type='Request',
                id=id
            )
        )

        while True:
            if length >= self.CONFIG["PROCESS_MAX_LENGTH"] or\
            depth >= self.CONFIG["PROCESS_MAX_DEPTH"] or\
            random.random() < self.CONFIG["REQUEST_END_CHANCE"]/100:
                process.append(
                    self.EventDto(
                        source=target,
                        target=source,
                        time=-1,
                        type='Response',
                        id=id
                    )
                )
                return length
            else: 
                length = self.add_process_event(
                    target, 
                    random.choice(tuple(servers)),
                    id,
                    depth+1,
                    length+2,
                    process,
                    servers
                )


    def random_milliseconds(self, start=False):
        if start:
            return 1000
        else:
            return random.randrange(
                *random.choices(
                    self.REQUEST_TIME_RANGES,
                    weights = self.REQUEST_TIME_WEIGHTS
                )[0]
            )
    

    def generate_process_patterns(self, pattern_number=None, save=True):
        
        if not pattern_number:
            pattern_number = self.CONFIG["PROCESS_PATTERN_NUMBER"]
       
        id = -1

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

        if save:
            file_path = "./output/patterns.csv"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as f:
                for pattern in self.patterns:
                    for event in pattern:
                        f.write(
                            f"<{event.source},{event.target},{event.time},{event.type},{event.id}>\n"
                        )


    def populate_processes(self, process_number=None, save=True):
        
        if not process_number:
            process_number = self.CONFIG["PROCESSES_TO_GENERATE"]
        
        log = []
        start_time = round(time.time()*1000)
        id = 0

        if save:
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
                    event.target=random.choice(tuple(self.SERVER_DEFINITIONS[event.target]["instances"]))
                                           
                    resp.source = event.target
                    resp.target = event.source

                if event.target:
                    next_req = pattern_copy[i+1]
                    next_req.source = event.target

                duration = event_time - start_time
                event_time += self.random_milliseconds()

            log.extend(pattern_copy)

            with open(stats_path, "a") as f:
                f.write(f"{id},{pattern[0].id},{duration},{[x.target for x in pattern_copy if x.type == 'Request']}\n")
            id += 1

        if save:
            file_path = "./output/unsorted_log.csv"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as f:
                for event in log:
                    f.write(f"<{event.source},{event.target},{event.time},{event.type},{event.id}>\n")

        sorted_log = sorted(log, key=lambda x: x.time)

        if save:
            file_path = "./output/log.csv"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as f:
                for event in sorted_log:
                    f.write(f"<{event.source},{event.target},{event.time},{event.type},{event.id}>\n")

        return sorted_log
    

def main():
    st = time.time()
    generator = Generator()
    generator._set_()
    generator.generate_process_patterns()
    generator.populate_processes()
    print("--- %s seconds ---" % (time.time() - st))


if __name__ == "__main__":
    main()