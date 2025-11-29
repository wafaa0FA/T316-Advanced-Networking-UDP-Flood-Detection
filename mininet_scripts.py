#!/usr/bin/env python3
"""
TMA Mininet Experiment — Fixed for PCAP copy timing

Key fix:
- Wait for tcpdump to finish AND copy files before stopping network
- Use proper synchronization to ensure files are copied
"""

import os
import sys
import time
import json
import signal
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

from mininet.net import Mininet
from mininet.node import Host, OVSSwitch
from mininet.link import TCLink
from mininet.util import dumpNodeConnections
from mininet.log import setLogLevel

# ---------------- CONFIG ----------------
BASE_DIR = Path("/home/ubuntu/tma_project")  # Fixed path instead of Path.home()
MAX_RETRIES = 2
RETRY_DELAY = 5  # seconds

# ports used by experiment (kept constant for clarity)
PORT_TCP = 52011
PORT_UDP_NORMAL = 52012
PORT_UDP_ATTACK = 52013

# iperf durations
NORMAL_DURATION_S = 30
ATTACK_DURATION_S = 15

# tcpdump capture tuning
MAX_CAPTURE_TIME = 40           # seconds
CAPTURE_SNAPLEN = 0             # full packet capture
CAPTURE_FILTER = ""             # capture all traffic

# global references
net: Optional[Mininet] = None
logger = None
tcpdump_pids = {}  # Track tcpdump PIDs for proper cleanup

# ---------------- Helpers ----------------
def check_root():
    if os.geteuid() != 0:
        print("ERROR: This script must be run as root (sudo).", file=sys.stderr)
        sys.exit(1)

def check_dependencies():
    reqs = ["tcpdump", "iperf3", "tc", "ovs-vsctl", "pgrep", "lsof", "tshark"]
    missing = [r for r in reqs if not shutil.which(r)]
    if missing:
        print("Missing dependencies:", ", ".join(missing), file=sys.stderr)
        sys.exit(1)
    # Ensure OVS running
    try:
        subprocess.run(["ovs-vsctl", "show"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception:
        print("OVS seems not running. Start with: sudo systemctl start openvswitch-switch", file=sys.stderr)
        sys.exit(1)

def setup_logging_and_dirs(exp_id: str):
    pcaps_dir = BASE_DIR / "pcaps" / exp_id
    pcaps_dir.mkdir(parents=True, exist_ok=True)
    log_file = pcaps_dir / "experiment.log"
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(), pcaps_dir

def run_host_bg(host: Host, cmd: str):
    """Run `cmd` in host background."""
    host.cmd(f"bash -c \"{cmd}\" > /dev/null 2>&1 &")
    logger.info("Started on %s: %s", host.name, cmd)

def host_pids_by_name(host: Host, name: str) -> List[int]:
    """Return list of PIDs inside host namespace matching process name (pgrep)."""
    res = host.cmd(f"pgrep -f '{name}'").strip()
    if not res:
        return []
    pids = []
    for token in res.split():
        try:
            pids.append(int(token))
        except ValueError:
            continue
    return pids

def kill_pids_in_host(host: Host, pids: List[int], sig: int = 15):
    for pid in pids:
        host.cmd(f"kill -{sig} {pid} 2>/dev/null || true")

def kill_by_name_in_host(host: Host, name: str, soft_first: bool = True):
    pids = host_pids_by_name(host, name)
    if not pids:
        return
    if soft_first:
        kill_pids_in_host(host, pids, sig=15)
        time.sleep(1)
        remaining = host_pids_by_name(host, name)
        if remaining:
            kill_pids_in_host(host, remaining, sig=9)
    else:
        kill_pids_in_host(host, pids, sig=9)
    logger.info("Killed processes named '%s' on %s (pids: %s)", name, host.name, pids)

def wait_for_port_in_host(host: Host, port: int, timeout: int = 10) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        res = host.cmd(f"timeout 0.5 bash -c 'echo > /dev/tcp/127.0.0.1/{port}' 2>/dev/null && echo OK || true")
        if "OK" in res:
            return True
        time.sleep(0.1)
    return False

def wait_for_file_exists(file_path: Path, timeout: int = 10) -> bool:
    """Wait for file to exist in controller filesystem."""
    start = time.time()
    while time.time() - start < timeout:
        if file_path.exists() and file_path.stat().st_size > 0:
            return True
        time.sleep(0.2)
    return False

def pcap_has_packets(pcap_path: Path) -> bool:
    if not pcap_path.exists():
        logger.warning("PCAP file does not exist: %s", pcap_path)
        return False
    
    file_size = pcap_path.stat().st_size
    if file_size < 24:  # PCAP header is 24 bytes
        logger.warning("PCAP file too small (%d bytes): %s", file_size, pcap_path)
        return False
    
    try:
        res = subprocess.run(
            ["tcpdump", "-r", str(pcap_path), "-n", "-c", "1"],
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True, 
            timeout=5
        )
        # If tcpdump can read at least one packet, it succeeds
        return res.returncode == 0
    except Exception as e:
        logger.warning("Error checking PCAP %s: %s", pcap_path, e)
        return False

def extract_pcap_counts(pcap_path: Path) -> Dict[str, int]:
    stats = {"total": 0, "tcp": 0, "udp": 0, "icmp": 0}
    if not pcap_path.exists():
        return stats
    try:
        res = subprocess.run(
            ["tcpdump", "-r", str(pcap_path), "-n", "-q"],
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True, 
            timeout=20
        )
        lines = res.stdout.splitlines()
        stats["total"] = len(lines)
        for line in lines:
            if "tcp" in line.lower():
                stats["tcp"] += 1
            elif "udp" in line.lower():
                stats["udp"] += 1
            elif "icmp" in line.lower():
                stats["icmp"] += 1
    except Exception as e:
        logger.warning("Error extracting stats from %s: %s", pcap_path, e)
    return stats

# ---------------- Phase Routines ----------------
def start_tcpdump_on_host(host: Host, pcaps_dir: Path, name: str, duration: int) -> Optional[str]:
    """
    Start tcpdump inside the host writing directly to controller filesystem.
    Returns the final filename or None on failure.
    """
    intf = host.defaultIntf().name
    final_path = pcaps_dir / name
    
    cap_time = min(duration, MAX_CAPTURE_TIME)
    
    # Write directly to the controller filesystem (accessible from host namespace)
    filter_str = CAPTURE_FILTER if CAPTURE_FILTER else ""
    snaplen_str = f"-s{CAPTURE_SNAPLEN}" if CAPTURE_SNAPLEN > 0 else "-s0"
    
    tcpdump_cmd = f"tcpdump -i {intf} -w {final_path} {snaplen_str} -U -nn {filter_str}"
    
    run_host_bg(host, tcpdump_cmd)
    
    # Store for cleanup
    time.sleep(0.5)
    pids = host_pids_by_name(host, f"tcpdump.*{intf}")
    if pids:
        tcpdump_pids[f"{host.name}_{name}"] = (host, pids[0])
    
    # Wait for file to appear
    if not wait_for_file_exists(final_path, timeout=5):
        logger.warning("tcpdump file did not appear: %s", final_path)
        return None
    
    logger.info("tcpdump started on %s -> %s for max %ds", host.name, final_path, cap_time)
    return final_path.name

def stop_all_tcpdumps():
    """Stop all tcpdump processes gracefully."""
    logger.info("Stopping all tcpdump processes...")
    for key, (host, pid) in tcpdump_pids.items():
        logger.info("Stopping tcpdump PID %d on %s", pid, host.name)
        host.cmd(f"kill -TERM {pid} 2>/dev/null || true")
    
    time.sleep(2)  # Give time for graceful shutdown
    
    # Force kill any remaining
    for key, (host, pid) in tcpdump_pids.items():
        host.cmd(f"kill -9 {pid} 2>/dev/null || true")
    
    time.sleep(1)  # Wait for filesystem to flush
    tcpdump_pids.clear()

def run_phase_normal(net: Mininet, pcaps_dir: Path, exp_tag: str) -> List[str]:
    h1, h2, h3, h4 = net.get('h1', 'h2', 'h3', 'h4')
    normal_pcaps: List[str] = []

    tcpdump_duration = NORMAL_DURATION_S + 10
    for host, fname in [(h1, "h1-normal.pcap"), (h2, "h2-normal.pcap"), 
                        (h3, "h3-normal.pcap"), (h4, "h4-normal.pcap")]:
        res = start_tcpdump_on_host(host, pcaps_dir, fname, tcpdump_duration)
        if res:
            normal_pcaps.append(res)

    time.sleep(2)

    # Start servers
    run_host_bg(h2, f"iperf3 -s -p {PORT_TCP}")
    run_host_bg(h4, f"iperf3 -s -p {PORT_UDP_NORMAL}")

    if not wait_for_port_in_host(h2, PORT_TCP, timeout=8):
        raise RuntimeError("TCP server on h2 did not bind in time")
    if not wait_for_port_in_host(h4, PORT_UDP_NORMAL, timeout=8):
        raise RuntimeError("UDP server on h4 did not bind in time")

    logger.info("Servers are ready, starting clients...")

    # Start clients
    run_host_bg(h1, f"iperf3 -c {h2.IP()} -p {PORT_TCP} -t {NORMAL_DURATION_S}")
    run_host_bg(h3, f"iperf3 -c {h4.IP()} -p {PORT_UDP_NORMAL} -u -b 20M -t {NORMAL_DURATION_S}")
    run_host_bg(h1, f"ping -c {NORMAL_DURATION_S//2} {h4.IP()}")

    logger.info("Waiting for normal traffic to complete (%d seconds)...", NORMAL_DURATION_S)
    time.sleep(NORMAL_DURATION_S + 5)

    # Kill servers gracefully
    kill_by_name_in_host(h2, "iperf3")
    kill_by_name_in_host(h4, "iperf3")
    kill_by_name_in_host(h1, "ping")

    logger.info("Normal phase completed")
    return normal_pcaps

def run_phase_attack(net: Mininet, pcaps_dir: Path, exp_tag: str) -> List[str]:
    h1, h3 = net.get('h1', 'h3')
    attack_pcaps: List[str] = []

    tcpdump_duration = ATTACK_DURATION_S + 10
    for host, fname in [(h1, "h1-attack.pcap"), (h3, "h3-attack.pcap")]:
        res = start_tcpdump_on_host(host, pcaps_dir, fname, tcpdump_duration)
        if res:
            attack_pcaps.append(res)

    time.sleep(2)

    # start server then client
    run_host_bg(h3, f"iperf3 -s -p {PORT_UDP_ATTACK}")
    if not wait_for_port_in_host(h3, PORT_UDP_ATTACK, timeout=8):
        raise RuntimeError("Attack server (h3) not ready")

    logger.info("Attack server ready, starting attack client...")
    run_host_bg(h1, f"iperf3 -c {h3.IP()} -p {PORT_UDP_ATTACK} -u -b 100M -t {ATTACK_DURATION_S}")

    logger.info("Waiting for attack traffic to complete (%d seconds)...", ATTACK_DURATION_S)
    time.sleep(ATTACK_DURATION_S + 5)

    # kill attack server/client
    kill_by_name_in_host(h3, "iperf3")
    kill_by_name_in_host(h1, "iperf3")

    logger.info("Attack phase completed")
    return attack_pcaps

# ---------------- Cleanup & main attempt ----------------
def force_cleanup_namespace():
    global net
    try:
        if net:
            net.stop()
    except Exception:
        pass
    try:
        subprocess.run(["pkill", "-9", "-f", "iperf3"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["pkill", "-9", "-f", "tcpdump"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    try:
        subprocess.run(["mn", "-c"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
    except Exception:
        pass
    time.sleep(2)

def run_single_attempt(exp_id: str) -> bool:
    global net, logger, tcpdump_pids
    tcpdump_pids.clear()
    
    logger, pcaps_dir = setup_logging_and_dirs(exp_id)
    logger.info("Starting attempt %s", exp_id)

    # create and start topology
    try:
        net = Mininet(host=Host, switch=OVSSwitch, link=TCLink, controller=None, autoStaticArp=True)
        h1 = net.addHost('h1', ip='10.0.0.1/24')
        h2 = net.addHost('h2', ip='10.0.0.2/24')
        h3 = net.addHost('h3', ip='10.0.0.3/24')
        h4 = net.addHost('h4', ip='10.0.0.4/24')
        s1 = net.addSwitch('s1', dpid='0000000000000001', failMode='standalone', inNamespace=False)
        for h in (h1, h2, h3, h4):
            net.addLink(h, s1, bw=10, delay='5ms', max_queue_size=1000)
        net.start()
        dumpNodeConnections(net.hosts)
        logger.info("Topology started")
    except Exception as e:
        logger.exception("Failed to create/start topology: %s", e)
        force_cleanup_namespace()
        return False

    normal_pcaps = []
    attack_pcaps = []
    success = False

    try:
        normal_pcaps = run_phase_normal(net, pcaps_dir, exp_id)
        logger.info("Waiting before attack phase...")
        time.sleep(3)
        attack_pcaps = run_phase_attack(net, pcaps_dir, exp_id)

        # CRITICAL: Stop tcpdump before stopping network
        logger.info("Stopping tcpdump processes...")
        stop_all_tcpdumps()
        
        # Additional wait to ensure all files are flushed
        logger.info("Waiting for files to flush...")
        time.sleep(3)

        verified_normal = []
        verified_attack = []
        
        for p in normal_pcaps:
            pcap_path = pcaps_dir / p
            if pcap_has_packets(pcap_path):
                verified_normal.append(p)
                logger.info("Verified normal PCAP: %s (size: %d bytes)", p, pcap_path.stat().st_size)
            else:
                logger.warning("Normal PCAP has no packets: %s", p)
        
        for p in attack_pcaps:
            pcap_path = pcaps_dir / p
            if pcap_has_packets(pcap_path):
                verified_attack.append(p)
                logger.info("Verified attack PCAP: %s (size: %d bytes)", p, pcap_path.stat().st_size)
            else:
                logger.warning("Attack PCAP has no packets: %s", p)

        logger.info("Verified normal pcaps: %s", verified_normal)
        logger.info("Verified attack pcaps: %s", verified_attack)

        metadata = {
            "experiment_id": exp_id,
            "start_time": datetime.now().isoformat(),
            "end_time": datetime.now().isoformat(),
            "pcap_files": {"normal": verified_normal, "attack": verified_attack},
            "ports": {"tcp": PORT_TCP, "udp_normal": PORT_UDP_NORMAL, "udp_attack": PORT_UDP_ATTACK},
            "notes": "Link bw set to 10 Mbps in TC; attack requested 100 Mbps"
        }
        with open(pcaps_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        for p in verified_normal:
            stats = extract_pcap_counts(pcaps_dir / p)
            logger.info("PCAP %s stats: %s", p, stats)
        for p in verified_attack:
            stats = extract_pcap_counts(pcaps_dir / p)
            logger.info("PCAP %s stats: %s", p, stats)

        success = bool(verified_normal and verified_attack)
        if not success:
            logger.warning("One or more PCAPs have zero packets (possible failure)")

    except Exception as e:
        logger.exception("Experiment runtime error: %s", e)
        success = False
    finally:
        try:
            force_cleanup_namespace()
        except Exception:
            pass

    return success

# ---------------- Main runner ----------------
def main():
    check_root()
    check_dependencies()
    setLogLevel('info')

    # install simple signal handlers
    def handle_sig(signum, frame):
        print("\nSignal received, cleaning up...")
        try:
            stop_all_tcpdumps()
            force_cleanup_namespace()
        finally:
            sys.exit(1)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, handle_sig)

    for attempt in range(MAX_RETRIES + 1):
        exp_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_try{attempt}"
        print(f"\n=== Attempt {attempt+1}/{MAX_RETRIES+1} ID: {exp_id} ===")
        ok = run_single_attempt(exp_id)
        if ok:
            print("Experiment succeeded.")
            return
        else:
            print("Experiment failed on attempt", attempt+1)
            if attempt < MAX_RETRIES:
                print(f"Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)

    print("All attempts failed.")
    sys.exit(1)

if __name__ == "__main__":
    main()
