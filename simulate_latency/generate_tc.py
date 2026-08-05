from latency_config import *


def generate_node_script(src):

    lines = []
    src_ip = NODES[src]

    lines.append("#!/bin/bash\n")
    lines.append("set -e\n\n")

    # Auto-detect the interface that owns this node's experiment IP.
    # Different CloudLab machines use different NIC names (enp3s0f0, enp94s0f1, ...).
    lines.append(
        f"""IFACE=$(ip -4 -o addr show to {src_ip} | awk '{{print $2; exit}}')
if [ -z "$IFACE" ]; then
  echo "Cannot find interface for {src_ip}" >&2
  exit 1
fi
echo "Using interface $IFACE for {src_ip}"

"""
    )

    iface = "$IFACE"

    # clear old rules
    lines.append(
        f"sudo tc qdisc del dev {iface} root 2>/dev/null || true\n\n"
    )

    # create root
    lines.append(
        f"sudo tc qdisc add dev {iface} root handle 1: prio\n\n"
    )

    band = 1

    for dst, delay in LATENCY[src].items():

        if src == dst:
            continue

        dst_ip = NODES[dst]

        #
        # create netem
        #

        lines.append(
            f"""
sudo tc qdisc add dev {iface} \\
    parent 1:{band} \\
    handle {band}0: \\
    netem delay {delay}ms

"""
        )

        #
        # create filter
        #

        lines.append(
            f"""
sudo tc filter add dev {iface} \\
    protocol ip \\
    parent 1:0 \\
    prio {band} \\
    u32 match ip dst {dst_ip} \\
    flowid 1:{band}

"""
        )

        band += 1

    return "".join(lines)


if __name__ == "__main__":

    for node in NODES:

        filename = f"{node}_tc.sh"

        with open(filename, "w") as f:
            f.write(generate_node_script(node))

        print(
            f"Generated {filename}"
        )
