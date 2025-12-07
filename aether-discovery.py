#!/usr/bin/env python3
"""
AETHER Discovery Service - Handles both registration and heartbeat packets
"""
import socket
import json
import time
import subprocess
from datetime import datetime, timedelta

DISCOVERY_PORT = 9999
DISCOVERED_FILE = "/home/ramzt/aether-discovered.json"
STALE_TIMEOUT = 120  # Remove nodes not seen in 2 minutes

def load_discovered():
    """Load discovered nodes from JSON file"""
    try:
        with open(DISCOVERED_FILE, 'r') as f:
            return json.load(f)
    except:
        return {}

def save_discovered(nodes):
    """Save discovered nodes to JSON file"""
    with open(DISCOVERED_FILE, 'w') as f:
        json.dump(nodes, f, indent=2)

def get_connected_macs():
    """Get MAC addresses of actually connected WiFi clients"""
    try:
        result = subprocess.run(['iw', 'dev', 'wlan0', 'station', 'dump'],
                              capture_output=True, text=True)
        macs = []
        for line in result.stdout.split('\n'):
            if line.startswith('Station '):
                mac = line.split()[1].upper()
                macs.append(mac)
        return macs
    except Exception as e:
        print(f"⚠️  Failed to get connected MACs: {e}")
        return []

def cleanup_stale_nodes(nodes):
    """Remove nodes that haven't been seen recently OR aren't connected to WiFi"""
    connected_macs = get_connected_macs()
    now = datetime.now()
    stale_timeout = timedelta(seconds=STALE_TIMEOUT)

    cleaned = {}
    removed = []

    for node_id, node in nodes.items():
        last_seen = datetime.fromisoformat(node.get('last_seen', node.get('first_seen')))
        mac = node.get('mac', '').upper()

        # Keep if: recently seen AND connected to WiFi
        if (now - last_seen) < stale_timeout and mac in connected_macs:
            cleaned[node_id] = node
        else:
            removed.append(node.get('assigned_name', node_id))

    if removed:
        print(f"🗑️  Removed stale nodes: {', '.join(removed)}")

    return cleaned

def handle_registration(msg, addr):
    """Handle registration packet from ESP32"""
    # Convert node_id to string for use as key
    node_id = str(msg.get('node_id') or msg.get('nodeId', ''))
    
    if not node_id:
        print(f"⚠️  Registration missing node_id from {addr[0]}")
        return None
    
    hostname = msg.get('hostname') or msg.get('nodeName', f'Node-{node_id[:8]}')
    mac = msg.get('mac', '').upper()
    universe = msg.get('universe', 1)
    start_channel = msg.get('startChannel') or msg.get('start_channel', 1)
    channel_count = msg.get('channelCount') or msg.get('channel_count', 512)
    
    nodes = load_discovered()
    
    if node_id not in nodes:
        nodes[node_id] = {
            'node_id': node_id,
            'assigned_name': hostname,
            'mac': mac,
            'universe': universe,
            'startChannel': start_channel,
            'channelCount': channel_count,
            'first_seen': datetime.now().isoformat()
        }
        print(f"📡 Discovered: {hostname} ({addr[0]}) Universe:{universe} Ch:{start_channel}-{start_channel+channel_count-1}")
    else:
        # Update existing node info
        nodes[node_id]['assigned_name'] = hostname
        nodes[node_id]['universe'] = universe
        nodes[node_id]['startChannel'] = start_channel
        nodes[node_id]['channelCount'] = channel_count
    
    # Update last_seen and IP
    nodes[node_id]['last_seen'] = datetime.now().isoformat()
    nodes[node_id]['ip'] = addr[0]
    
    save_discovered(nodes)
    return node_id

def handle_heartbeat(msg, addr):
    """Handle heartbeat packet from ESP32"""
    # Convert node_id to string for use as key
    node_id = str(msg.get('node_id', ''))
    
    if not node_id:
        print(f"⚠️  Heartbeat missing node_id from {addr[0]}")
        return None
    
    nodes = load_discovered()
    
    if node_id in nodes:
        # Update last_seen
        nodes[node_id]['last_seen'] = datetime.now().isoformat()
        nodes[node_id]['ip'] = addr[0]
        
        # Optionally update stats
        if 'uptime' in msg:
            nodes[node_id]['uptime'] = msg['uptime']
        if 'rssi' in msg:
            nodes[node_id]['rssi'] = msg['rssi']
        if 'fps' in msg:
            nodes[node_id]['fps'] = msg['fps']
        
        save_discovered(nodes)
    else:
        # Node sent heartbeat but isn't registered - might be a re-registration
        print(f"⚠️  Heartbeat from unknown node {node_id} at {addr[0]} - waiting for registration")
    
    return node_id

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('', DISCOVERY_PORT))
    sock.settimeout(10)  # 10 second timeout for cleanup cycle

    print("=" * 60)
    print("AETHER Discovery Service")
    print("=" * 60)
    print(f"✓ Listening on port {DISCOVERY_PORT}")
    print(f"✓ Auto-cleanup: nodes offline >{STALE_TIMEOUT}s removed")
    print("=" * 60)

    last_cleanup = time.time()

    while True:
        try:
            # Try to receive discovery packet
            data, addr = sock.recvfrom(1024)
            msg = json.loads(data.decode())
            
            packet_type = msg.get('type')
            
            if packet_type == 'register':
                handle_registration(msg, addr)
            elif packet_type == 'heartbeat':
                handle_heartbeat(msg, addr)
            else:
                print(f"⚠️  Unknown packet type '{packet_type}' from {addr[0]}")

        except socket.timeout:
            pass  # Normal timeout for cleanup cycle
        except json.JSONDecodeError as e:
            print(f"⚠️  Invalid JSON from {addr[0]}: {e}")
        except Exception as e:
            print(f"⚠️  Discovery error: {e}")
            import traceback
            traceback.print_exc()

        # Cleanup stale nodes every 30 seconds
        if time.time() - last_cleanup > 30:
            nodes = load_discovered()
            cleaned = cleanup_stale_nodes(nodes)
            if len(cleaned) != len(nodes):
                save_discovered(cleaned)
            last_cleanup = time.time()

if __name__ == '__main__':
    main()
