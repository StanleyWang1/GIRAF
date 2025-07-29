import csv
import time
from utils.NatNetClient import NatNetClient

def stream_and_log(client_ip, server_ip, table_id, gripper_id, csv_filename):
    """
    Starts streaming from Motive server and logs pose data for table and gripper to CSV.
    """
    TARGET_IDS = {table_id, gripper_id}

    csv_file = open(csv_filename, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["timestamp", "rigid_body_id", "x", "y", "z", "qx", "qy", "qz", "qw"])

    def receive_rigid_body_frame(rigid_body_id, position, quaternion):
        if rigid_body_id not in TARGET_IDS:
            return
        timestamp = time.time()
        writer.writerow([timestamp, rigid_body_id, *position, *quaternion])
        print(f"[{timestamp:.2f}] RB{rigid_body_id}  pos={position}  quat={quaternion}")

    client = NatNetClient()
    client.set_client_address(client_ip)
    client.set_server_address(server_ip)
    client.set_use_multicast(False)
    client.rigid_body_listener = receive_rigid_body_frame

    if not client.run("d"):
        print("❌ Failed to connect to Motive server.")
        return

    print(f"✅ Streaming data for Rigid Bodies: TABLE={table_id}, GRIPPER={gripper_id}")
    print("📦 Saving to:", csv_filename)
    print("🛑 Press Ctrl+C to stop recording.")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("🛑 Stopping...")
        client.shutdown()
        csv_file.close()
        print("✅ CSV saved and client shut down.")
