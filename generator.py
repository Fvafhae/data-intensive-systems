import time
import random

import numpy as np
import pandas as pd


MAX_DEPTH = 3
MAX_LENGTH = 20

CONFIG = {
    "ranges": {
        "normal": (100, 800),
        "mid": (801, 2000),
        "long": (2001, 10000)
    },
    "weights": {
        "normal": 75,
        "mid": 20,
        "long": 5
    },
}

RANGES = [CONFIG["ranges"][key] for key in CONFIG["ranges"].keys()]
WEIGHTS = [CONFIG["weights"][key] for key in CONFIG["weights"].keys()]

print(random.choices(
        RANGES,
        weights = WEIGHTS
    ))
open('log.csv', 'w').close()
open('stats.csv', 'w').close()

# TODO: Random generation of Servers
SERVERS = set([1,2,3,4,5,6,7,8,9,10])
id = -1


def gen(caller, target, time, id, depth, length, log, servers):

    servers = set(servers) - set([target])
    time = append_event(
        caller, 
        target, 
        time,
        'Request',
        id,
        log
    )

    while True:
        if length >= MAX_LENGTH or\
        depth >= MAX_DEPTH or\
        random.random() < 0.2:
            time = append_event(
                target,
                caller,
                time,
                'Response',
                id,
                log
            )
            return time, length
        else: 
            time, length = gen(
                target, 
                random.choice(tuple(servers)),
                time + random_milliseconds(),
                id,
                depth+1,
                length+2,
                log,
                servers
            )

def append_event(caller, target, time, type, id, log):

    time += random_milliseconds()
    log.append(
        (
            caller,
            target,
            time,
            type,
            id
        )
    )
    return time

def random_milliseconds():
    
    return random.randrange(
        *random.choices(
            RANGES, 
            weights = WEIGHTS
        )[0]
    )

start_time = round(time.time()*1000)

LOG = []

for _ in range(2):

    id += 1
    events = []

    # TODO: Save events in another file
    # TODO: Merge events in one file
    end_time, length = gen(None, random.choice(tuple(SERVERS)), start_time, id, 0, 2, events, SERVERS)
    dur = end_time - start_time 
    start_time += random_milliseconds()
    LOG.extend(events)
    with open("stats.csv", "a") as f:
        f.write(f"{id},{length},{dur}\n")

with open("log.csv", "a") as f:
    for event in LOG:
        f.write(f"{event}\n")

print(events)