#!/usr/bin/env python3
"""
Admin operations example for Dakera Python SDK.

This example demonstrates:
- Creating and managing backups
- Configuring quotas
- Maintenance mode management
- Cluster status and replication info
"""

import os
import sys

from dakera import DakeraClient


def main():
    # Connect to Dakera server
    client = DakeraClient(
        os.environ.get("DAKERA_API_URL", "http://localhost:3300"),
        api_key=os.environ.get("DAKERA_API_KEY", "dk-mykey"),
    )

    # Check server health
    health = client.health()
    print(f"Server status: {health}")
    assert health is not None, "expected non-None health response"

    # --- Backup Operations ---
    print("\n=== Backup Operations ===")

    # Create a full backup
    backup = client.admin_create_backup(include_data=True, description="example backup")
    print(f"Created backup: {backup}")
    assert backup is not None, "expected non-None backup result"
    backup_id = backup.get("backup_id") or backup.get("id")

    # List all backups
    backups = client.admin_list_backups()
    print(f"Total backups: {len(backups) if isinstance(backups, list) else backups}")

    # Get backup schedule
    schedule = client.admin_get_backup_schedule()
    print(f"Backup schedule: {schedule}")

    # Update backup schedule (daily at 2am)
    updated_schedule = client.admin_update_backup_schedule(
        enabled=True, cron="0 2 * * *", retention_days=30
    )
    print(f"Updated schedule: {updated_schedule}")

    # Delete test backup
    if backup_id:
        client.admin_delete_backup(backup_id)
        print(f"Deleted backup: {backup_id}")

    # --- Quota Management ---
    print("\n=== Quota Management ===")

    # Get default quota configuration
    default_quota = client.admin_get_default_quota()
    print(f"Default quota: {default_quota}")
    assert default_quota is not None, "expected non-None default quota"

    # Set quota for a specific namespace
    quota_config = {
        "max_vectors": 100000,
        "max_storage_bytes": 1073741824,  # 1 GB
        "max_queries_per_minute": 1000,
    }
    client.admin_set_quota("example-namespace", quota_config)
    print(f"Set quota for example-namespace: {quota_config}")

    # Check quota usage
    quota_check = client.admin_check_quota("example-namespace")
    print(f"Quota check: {quota_check}")

    # Clean up quota
    client.admin_delete_quota("example-namespace")
    print("Deleted quota for example-namespace")

    # --- Maintenance Mode ---
    print("\n=== Maintenance Mode ===")

    # Check current maintenance status
    maint_status = client.admin_maintenance_status()
    print(f"Maintenance status: {maint_status}")
    assert maint_status is not None, "expected non-None maintenance status"

    # Enable maintenance mode with a reason
    enable_resp = client.admin_enable_maintenance(reason="Scheduled maintenance window")
    print(f"Enabled maintenance: {enable_resp}")

    # Disable maintenance mode
    disable_resp = client.admin_disable_maintenance()
    print(f"Disabled maintenance: {disable_resp}")

    # --- Cluster Operations ---
    print("\n=== Cluster Operations ===")

    # Get cluster status
    cluster = client.cluster_status()
    print(f"Cluster status: {cluster}")
    assert cluster is not None, "expected non-None cluster status"

    # Get cluster nodes
    nodes = client.cluster_nodes()
    print(f"Cluster nodes ({len(nodes)}):")
    for node in nodes:
        print(f"  - {node}")

    # Get replication info
    replication = client.admin_cluster_replication()
    print(f"Replication info: {replication}")

    # List shards
    shards = client.admin_list_shards()
    print(f"Shards: {shards}")

    # --- Slow Query Management ---
    print("\n=== Slow Query Management ===")

    slow_summary = client.admin_slow_query_summary()
    print(f"Slow query summary: {slow_summary}")

    slow_queries = client.admin_list_slow_queries(limit=5)
    print(f"Recent slow queries: {slow_queries}")

    client.close()
    print("\nDone!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)
