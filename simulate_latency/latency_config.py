# latency_config.py

NODES = {
    "node0": "10.10.1.1",
    "node1": "10.10.1.2",
    "node2": "10.10.1.3",
    "node3": "10.10.1.4",
}

LATENCY = {
    "node0": {
        "node1": 100,
        "node2": 75,
        "node3": 60,
    },

    "node1": {
        "node0": 100,
        "node2": 20,
        "node3": 80,
    },

    "node2": {
        "node0": 75,
        "node1": 20,
        "node3": 100,
    },

    "node3": {
        "node0": 60,
        "node1": 80,
        "node2": 100,
    },
}
