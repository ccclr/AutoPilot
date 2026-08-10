# latency_config.py

NODES = {
    "node0": "10.10.1.1",
    "node1": "10.10.1.2",
    "node2": "10.10.1.3",
    "node3": "10.10.1.4",
}

LATENCY = {
    "node0": {
        "node1": 50,
        "node2": 50,
        "node3": 60,
    },

    "node1": {
        "node0": 50,
        "node2": 30,
        "node3": 10,
    },

    "node2": {
        "node0": 50,
        "node1": 30,
        "node3": 10,
    },

    "node3": {
        "node0": 60,
        "node1": 10,
        "node2": 10,
    },
}
