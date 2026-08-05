# latency_config.py

NODES = {
    "node0": "10.10.1.1",
    "node1": "10.10.1.2",
    "node2": "10.10.1.3",
    "node3": "10.10.1.4",
}


# 单向 delay (ms)
# latency[src][dst]

LATENCY = {
    "node0": {
        "node1": 20,
        "node2": 20,
        "node3": 20,
    },

    "node1": {
        "node0": 30,
        "node2": 30,
        "node3": 30,
    },

    "node2": {
        "node0": 40,
        "node1": 40,
        "node3": 40,
    },

    "node3": {
        "node0": 40,
        "node1": 40,
        "node2": 40,
    },
}
