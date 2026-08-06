from fabric import Connection
from latency_config import NODES


USER = "AlanXiao"
KEY_FILENAME = "/users/AlanXiao/.ssh/cloudlab"


def deploy(node):

    conn = Connection(
        host=NODES[node],
        user=USER,
        connect_kwargs={
            "key_filename": KEY_FILENAME,
        },
    )


    script=f"{node}_tc.sh"


    conn.put(
        script,
        "/tmp/tc_latency.sh"
    )


    conn.run(
        "chmod +x /tmp/tc_latency.sh"
    )


    conn.run(
        "sudo /tmp/tc_latency.sh"
    )


    print(
        f"{node} configured"
    )



for node in NODES:
    deploy(node)