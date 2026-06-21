"""Tests for admin methods."""

import json

import pytest
import responses

from dakera import DakeraClient, ServerError


@pytest.fixture
def client():
    """Create a test client."""
    return DakeraClient("http://localhost:3000")


@pytest.fixture
def mock_responses():
    """Enable responses mocking."""
    with responses.RequestsMock() as rsps:
        yield rsps


class TestClusterAdmin:
    """Tests for cluster administration methods."""

    def test_cluster_status(self, client, mock_responses):
        """Test getting cluster status."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/admin/cluster/status",
            json={"status": "healthy", "nodes": 3, "leader": "node-1"},
            status=200,
        )
        result = client.cluster_status()
        assert result["status"] == "healthy"
        assert result["nodes"] == 3
        assert len(mock_responses.calls) == 1

    def test_cluster_status_error(self, client, mock_responses):
        """Test cluster status with server error."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/admin/cluster/status",
            json={"error": "cluster unavailable"},
            status=500,
        )
        with pytest.raises(ServerError):
            client.cluster_status()

    def test_cluster_nodes(self, client, mock_responses):
        """Test getting cluster nodes."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/admin/cluster/nodes",
            json=[
                {"id": "node-1", "role": "leader", "status": "active"},
                {"id": "node-2", "role": "follower", "status": "active"},
            ],
            status=200,
        )
        result = client.cluster_nodes()
        assert len(result) == 2
        assert result[0]["id"] == "node-1"

    def test_admin_cluster_replication(self, client, mock_responses):
        """Test getting cluster replication status."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/admin/cluster/replication",
            json={"replication_factor": 3, "healthy_replicas": 3, "lag_ms": 12},
            status=200,
        )
        result = client.admin_cluster_replication()
        assert result["replication_factor"] == 3
        assert result["healthy_replicas"] == 3

    def test_admin_list_shards(self, client, mock_responses):
        """Test listing shards."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/admin/cluster/shards",
            json={"shards": [{"id": "shard-0", "size_bytes": 1024000}]},
            status=200,
        )
        result = client.admin_list_shards()
        assert "shards" in result

    def test_admin_rebalance_shards(self, client, mock_responses):
        """Test rebalancing shards."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/admin/cluster/shards/rebalance",
            json={"moved": 2, "total_shards": 8},
            status=200,
        )
        result = client.admin_rebalance_shards(shard_ids=["shard-0"], dry_run=True)
        assert result["moved"] == 2
        req_body = json.loads(mock_responses.calls[0].request.body)
        assert req_body["dry_run"] is True
        assert req_body["shard_ids"] == ["shard-0"]

    def test_admin_rebalance_shards_error(self, client, mock_responses):
        """Test rebalance with server error."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/admin/cluster/shards/rebalance",
            json={"error": "rebalance in progress"},
            status=500,
        )
        with pytest.raises(ServerError):
            client.admin_rebalance_shards()


class TestMaintenanceMode:
    """Tests for maintenance mode methods."""

    def test_admin_maintenance_status(self, client, mock_responses):
        """Test getting maintenance status."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/admin/cluster/maintenance",
            json={"enabled": False, "reason": None, "since": None},
            status=200,
        )
        result = client.admin_maintenance_status()
        assert result["enabled"] is False

    def test_admin_enable_maintenance(self, client, mock_responses):
        """Test enabling maintenance mode."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/admin/cluster/maintenance/enable",
            json={"enabled": True, "reason": "planned upgrade"},
            status=200,
        )
        result = client.admin_enable_maintenance(
            reason="planned upgrade",
            node_ids=["node-1"],
            reject_requests=True,
            duration_minutes=30,
        )
        assert result["enabled"] is True
        req_body = json.loads(mock_responses.calls[0].request.body)
        assert req_body["reason"] == "planned upgrade"
        assert req_body["reject_requests"] is True
        assert req_body["duration_minutes"] == 30
        assert req_body["node_ids"] == ["node-1"]

    def test_admin_disable_maintenance(self, client, mock_responses):
        """Test disabling maintenance mode."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/admin/cluster/maintenance/disable",
            json={"enabled": False},
            status=200,
        )
        result = client.admin_disable_maintenance(force=True)
        assert result["enabled"] is False
        req_body = json.loads(mock_responses.calls[0].request.body)
        assert req_body["force"] is True


class TestQuotaAdmin:
    """Tests for quota administration methods."""

    def test_admin_list_quotas(self, client, mock_responses):
        """Test listing all quotas."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/admin/quotas",
            json={"quotas": [{"namespace": "ns-1", "max_vectors": 100000}]},
            status=200,
        )
        result = client.admin_list_quotas()
        assert "quotas" in result

    def test_admin_get_default_quota(self, client, mock_responses):
        """Test getting default quota."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/admin/quotas/default",
            json={"max_vectors": 1000000, "max_namespaces": 100},
            status=200,
        )
        result = client.admin_get_default_quota()
        assert result["max_vectors"] == 1000000

    def test_admin_set_default_quota(self, client, mock_responses):
        """Test setting default quota."""
        mock_responses.add(
            responses.PUT,
            "http://localhost:3000/v1/admin/quotas/default",
            json={"max_vectors": 500000},
            status=200,
        )
        result = client.admin_set_default_quota({"max_vectors": 500000})
        assert result["max_vectors"] == 500000

    def test_admin_get_quota(self, client, mock_responses):
        """Test getting namespace quota."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/admin/quotas/test-ns",
            json={"namespace": "test-ns", "max_vectors": 50000},
            status=200,
        )
        result = client.admin_get_quota("test-ns")
        assert result["namespace"] == "test-ns"

    def test_admin_set_quota(self, client, mock_responses):
        """Test setting namespace quota."""
        mock_responses.add(
            responses.PUT,
            "http://localhost:3000/v1/admin/quotas/test-ns",
            json={"namespace": "test-ns", "max_vectors": 75000},
            status=200,
        )
        result = client.admin_set_quota("test-ns", {"max_vectors": 75000})
        assert result["max_vectors"] == 75000

    def test_admin_delete_quota(self, client, mock_responses):
        """Test deleting namespace quota."""
        mock_responses.add(
            responses.DELETE,
            "http://localhost:3000/v1/admin/quotas/test-ns",
            json={"deleted": True},
            status=200,
        )
        result = client.admin_delete_quota("test-ns")
        assert result["deleted"] is True

    def test_admin_check_quota(self, client, mock_responses):
        """Test checking if operation exceeds quota."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/admin/quotas/test-ns/check",
            json={"allowed": True, "remaining_vectors": 49000},
            status=200,
        )
        result = client.admin_check_quota(
            "test-ns", vector_ids=["v1", "v2"], dimensions=384, metadata_bytes=1024
        )
        assert result["allowed"] is True
        req_body = json.loads(mock_responses.calls[0].request.body)
        assert req_body["vector_ids"] == ["v1", "v2"]
        assert req_body["dimensions"] == 384


class TestSlowQueryAdmin:
    """Tests for slow query administration methods."""

    def test_admin_list_slow_queries(self, client, mock_responses):
        """Test listing slow queries."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/admin/slow-queries",
            json=[
                {"query_id": "q1", "duration_ms": 250, "namespace": "ns-1"},
                {"query_id": "q2", "duration_ms": 500, "namespace": "ns-2"},
            ],
            status=200,
        )
        result = client.admin_list_slow_queries(namespace="ns-1", limit=10)
        assert len(result) == 2
        assert result[0]["duration_ms"] == 250

    def test_admin_slow_query_summary(self, client, mock_responses):
        """Test getting slow query summary."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/admin/slow-queries/summary",
            json={"total": 15, "avg_duration_ms": 350, "p99_duration_ms": 800},
            status=200,
        )
        result = client.admin_slow_query_summary()
        assert result["total"] == 15
        assert result["avg_duration_ms"] == 350

    def test_admin_clear_slow_queries(self, client, mock_responses):
        """Test clearing slow query log."""
        mock_responses.add(
            responses.DELETE,
            "http://localhost:3000/v1/admin/slow-queries",
            json={"cleared": 15},
            status=200,
        )
        result = client.admin_clear_slow_queries(namespace="ns-1")
        assert result["cleared"] == 15

    def test_admin_update_slow_query_config(self, client, mock_responses):
        """Test updating slow query configuration."""
        mock_responses.add(
            responses.PATCH,
            "http://localhost:3000/v1/admin/slow-queries/config",
            json={"threshold_ms": 200, "max_entries": 500},
            status=200,
        )
        result = client.admin_update_slow_query_config(threshold_ms=200, max_entries=500)
        assert result["threshold_ms"] == 200


class TestBackupAdmin:
    """Tests for backup administration methods."""

    def test_admin_list_backups(self, client, mock_responses):
        """Test listing backups."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/admin/backups",
            json={
                "backups": [
                    {"id": "bak-1", "name": "daily-2026-05-17", "size_bytes": 5000000}
                ]
            },
            status=200,
        )
        result = client.admin_list_backups()
        assert "backups" in result

    def test_admin_create_backup(self, client, mock_responses):
        """Test creating a backup."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/admin/backups",
            json={"id": "bak-2", "name": "manual-backup", "status": "creating"},
            status=200,
        )
        result = client.admin_create_backup(
            name="manual-backup",
            backup_type="full",
            namespaces=["ns-1", "ns-2"],
            encrypt=True,
            compression="zstd",
        )
        assert result["id"] == "bak-2"
        req_body = json.loads(mock_responses.calls[0].request.body)
        assert req_body["name"] == "manual-backup"
        assert req_body["backup_type"] == "full"
        assert req_body["encrypt"] is True

    def test_admin_get_backup(self, client, mock_responses):
        """Test getting backup details."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/admin/backups/bak-1",
            json={"id": "bak-1", "name": "daily", "status": "complete", "size_bytes": 5000000},
            status=200,
        )
        result = client.admin_get_backup("bak-1")
        assert result["status"] == "complete"

    def test_admin_delete_backup(self, client, mock_responses):
        """Test deleting a backup."""
        mock_responses.add(
            responses.DELETE,
            "http://localhost:3000/v1/admin/backups/bak-1",
            json={"deleted": True},
            status=200,
        )
        result = client.admin_delete_backup("bak-1")
        assert result["deleted"] is True

    def test_admin_get_backup_schedule(self, client, mock_responses):
        """Test getting backup schedule."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/admin/backups/schedule",
            json={"cron": "0 2 * * *", "retention_days": 30, "enabled": True},
            status=200,
        )
        result = client.admin_get_backup_schedule()
        assert result["cron"] == "0 2 * * *"

    def test_admin_update_backup_schedule(self, client, mock_responses):
        """Test updating backup schedule."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/admin/backups/schedule",
            json={"cron": "0 3 * * *", "retention_days": 14, "enabled": True},
            status=200,
        )
        result = client.admin_update_backup_schedule(
            cron="0 3 * * *", retention_days=14, enabled=True
        )
        assert result["cron"] == "0 3 * * *"

    def test_admin_restore_backup(self, client, mock_responses):
        """Test restoring from backup."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/admin/backups/restore",
            json={"restore_id": "rst-1", "status": "in_progress"},
            status=200,
        )
        result = client.admin_restore_backup(
            "bak-1", target_namespaces=["ns-1"], overwrite=True
        )
        assert result["restore_id"] == "rst-1"
        req_body = json.loads(mock_responses.calls[0].request.body)
        assert req_body["backup_id"] == "bak-1"
        assert req_body["overwrite"] is True

    def test_admin_get_restore_status(self, client, mock_responses):
        """Test getting restore operation status."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/admin/backups/restore/rst-1",
            json={"restore_id": "rst-1", "status": "complete", "elapsed_ms": 5000},
            status=200,
        )
        result = client.admin_get_restore_status("rst-1")
        assert result["status"] == "complete"


class TestStorageAndTiers:
    """Tests for storage tier and memory type stats methods."""

    def test_storage_tier_overview(self, client, mock_responses):
        """Test getting storage tier overview."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/admin/storage/tiers",
            json={
                "tiers_enabled": True,
                "architecture": [
                    {
                        "name": "hot",
                        "tier_type": "memory",
                        "technology": "in-memory",
                        "description": "Hot tier for fast access",
                        "target_latency": "<1ms",
                        "capacity": "1GB",
                        "status": "active",
                        "current_count": 5000,
                        "hit_count": 4500,
                        "hit_rate": 0.9,
                    },
                    {
                        "name": "warm",
                        "tier_type": "disk",
                        "technology": "mmap",
                        "description": "Warm tier for moderate access",
                        "target_latency": "<10ms",
                        "capacity": "10GB",
                        "status": "active",
                        "current_count": 50000,
                        "hit_count": 30000,
                        "hit_rate": 0.6,
                    },
                ],
                "config": {
                    "hot_tier_capacity": 10000,
                    "hot_to_warm_threshold_secs": 3600,
                    "warm_to_cold_threshold_secs": 86400,
                    "auto_tier_enabled": True,
                    "tier_check_interval_secs": 300,
                },
                "activity": {
                    "promotions": 100,
                    "demotions": 50,
                    "cache_hit_rate": 0.85,
                    "storage_backend": "mmap",
                    "promotions_to_hot": 80,
                    "demotions_to_warm": 30,
                    "demotions_to_cold": 20,
                },
            },
            status=200,
        )
        result = client.storage_tier_overview()
        assert result.tiers_enabled is True
        assert len(result.architecture) == 2

    def test_storage_tier_overview_error(self, client, mock_responses):
        """Test storage tier overview with server error."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/admin/storage/tiers",
            json={"error": "storage unavailable"},
            status=500,
        )
        with pytest.raises(ServerError):
            client.storage_tier_overview()

    def test_memory_type_stats(self, client, mock_responses):
        """Test getting memory type distribution stats."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/admin/memory-type-stats",
            json={
                "total": 2550,
                "working": 50,
                "episodic": 1500,
                "semantic": 800,
                "procedural": 200,
                "agent_namespaces": 5,
            },
            status=200,
        )
        result = client.memory_type_stats()
        assert result.total == 2550
        assert result.working == 50

    def test_background_activity(self, client, mock_responses):
        """Test getting background activity."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/admin/background-activity",
            json={
                "tasks": [
                    {"name": "compaction", "status": "running", "progress": 0.5},
                    {"name": "decay", "status": "idle"},
                ]
            },
            status=200,
        )
        result = client.background_activity()
        assert len(result["tasks"]) == 2


class TestDimensionMigration:
    """Tests for dimension migration method."""

    def test_migrate_namespace_dimensions(self, client, mock_responses):
        """Test migrating namespace dimensions."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/admin/namespaces/migrate-dimensions",
            json={
                "migrated": 2,
                "failed": 0,
                "already_current": 0,
                "results": [
                    {
                        "namespace": "ns-1",
                        "original_dimension": 768,
                        "vectors_migrated": 100,
                        "vectors_skipped": 0,
                        "status": "completed",
                    },
                    {
                        "namespace": "ns-2",
                        "original_dimension": 768,
                        "vectors_migrated": 50,
                        "vectors_skipped": 0,
                        "status": "completed",
                    },
                ],
            },
            status=200,
        )
        result = client.migrate_namespace_dimensions(
            target_dimension=1024, namespaces=["ns-1", "ns-2"]
        )
        assert result.migrated == 2
        assert len(result.results) == 2
        req_body = json.loads(mock_responses.calls[0].request.body)
        assert req_body["target_dimension"] == 1024
        assert req_body["namespaces"] == ["ns-1", "ns-2"]

    def test_migrate_namespace_dimensions_error(self, client, mock_responses):
        """Test dimension migration with server error."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/admin/namespaces/migrate-dimensions",
            json={"error": "migration already in progress"},
            status=500,
        )
        with pytest.raises(ServerError):
            client.migrate_namespace_dimensions(target_dimension=1024)


class TestTTLAdmin:
    """Tests for TTL administration methods."""

    def test_configure_ttl(self, client, mock_responses):
        """Test configuring TTL for a namespace."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/admin/namespaces/test-ns/ttl",
            json={"namespace": "test-ns", "ttl_seconds": 86400, "strategy": "hard_delete"},
            status=200,
        )
        result = client.configure_ttl("test-ns", ttl_seconds=86400, strategy="hard_delete")
        assert result["ttl_seconds"] == 86400
        req_body = json.loads(mock_responses.calls[0].request.body)
        assert req_body["ttl_seconds"] == 86400
        assert req_body["strategy"] == "hard_delete"

    def test_ttl_stats(self, client, mock_responses):
        """Test getting TTL stats."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/admin/ttl/stats",
            json={
                "namespaces": [
                    {
                        "namespace": "ns-1",
                        "vectors_with_ttl": 100,
                        "expiring_within_hour": 5,
                        "expiring_within_day": 20,
                        "expired_pending_cleanup": 10,
                    }
                ],
                "total_with_ttl": 100,
                "total_expired": 10,
            },
            status=200,
        )
        result = client.ttl_stats()
        assert result.total_expired == 10
        assert result.total_with_ttl == 100


class TestEncryptionAdmin:
    """Tests for encryption key rotation."""

    def test_rotate_encryption_key(self, client, mock_responses):
        """Test rotating encryption key."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/admin/encryption/rotate-key",
            json={"rotated": 150, "skipped": 5, "namespaces": ["ns-1", "ns-2"]},
            status=200,
        )
        result = client.rotate_encryption_key(
            new_key="a" * 64, namespace="ns-1"
        )
        assert result.rotated == 150
        assert result.skipped == 5

    def test_rotate_encryption_key_error(self, client, mock_responses):
        """Test encryption rotation with server error."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/admin/encryption/rotate-key",
            json={"error": "invalid key format"},
            status=500,
        )
        with pytest.raises(ServerError):
            client.rotate_encryption_key(new_key="short")


class TestAutopilotAdmin:
    """Tests for autopilot administration methods."""

    def test_autopilot_status(self, client, mock_responses):
        """Test getting autopilot status."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/admin/autopilot/status",
            json={
                "enabled": True,
                "dedup_threshold": 0.9,
                "last_run": "2026-05-17T00:00:00Z",
            },
            status=200,
        )
        result = client.autopilot_status()
        assert result["enabled"] is True

    def test_autopilot_update_config(self, client, mock_responses):
        """Test updating autopilot config."""
        mock_responses.add(
            responses.PUT,
            "http://localhost:3000/v1/admin/autopilot/config",
            json={"enabled": False, "dedup_threshold": 0.85},
            status=200,
        )
        result = client.autopilot_update_config(enabled=False, dedup_threshold=0.85)
        assert result["enabled"] is False
        req_body = json.loads(mock_responses.calls[0].request.body)
        assert req_body["enabled"] is False
        assert req_body["dedup_threshold"] == 0.85

    def test_autopilot_trigger(self, client, mock_responses):
        """Test triggering autopilot action."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/admin/autopilot/trigger",
            json={"triggered": "dedup", "status": "started"},
            status=200,
        )
        result = client.autopilot_trigger("dedup")
        assert result["triggered"] == "dedup"


class TestDecayAdmin:
    """Tests for decay engine administration methods."""

    def test_decay_config(self, client, mock_responses):
        """Test getting decay config."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/admin/decay/config",
            json={
                "strategy": "exponential",
                "half_life_hours": 336.0,
                "min_importance": 0.01,
            },
            status=200,
        )
        result = client.decay_config()
        assert result["strategy"] == "exponential"
        assert result["half_life_hours"] == 336.0

    def test_decay_update_config(self, client, mock_responses):
        """Test updating decay config."""
        mock_responses.add(
            responses.PUT,
            "http://localhost:3000/v1/admin/decay/config",
            json={"strategy": "linear", "half_life_hours": 168.0, "min_importance": 0.05},
            status=200,
        )
        result = client.decay_update_config(
            strategy="linear", half_life_hours=168.0, min_importance=0.05
        )
        assert result["strategy"] == "linear"

    def test_decay_stats(self, client, mock_responses):
        """Test getting decay stats."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/admin/decay/stats",
            json={
                "total_decayed": 500,
                "total_deleted": 30,
                "cycles_run": 48,
                "last_cycle": {"decayed": 12, "deleted": 2},
            },
            status=200,
        )
        result = client.decay_stats()
        assert result["total_decayed"] == 500
        assert result["cycles_run"] == 48


class TestCacheAdmin:
    """Tests for cache administration methods."""

    def test_cache_stats(self, client, mock_responses):
        """Test getting cache stats."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/admin/cache/stats",
            json={"hits": 5000, "misses": 200, "hit_rate": 0.96, "size_bytes": 10000000},
            status=200,
        )
        result = client.cache_stats()
        assert result["hit_rate"] == 0.96

    def test_cache_clear(self, client, mock_responses):
        """Test clearing cache."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/admin/cache/clear",
            json={"cleared": True, "entries_evicted": 500},
            status=200,
        )
        result = client.cache_clear(namespace="test-ns")
        assert result["cleared"] is True


class TestConfigAdmin:
    """Tests for server configuration methods."""

    def test_get_config(self, client, mock_responses):
        """Test getting server config."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/admin/config",
            json={"max_vector_dim": 4096, "default_index": "hnsw"},
            status=200,
        )
        result = client.get_config()
        assert result["max_vector_dim"] == 4096

    def test_update_config(self, client, mock_responses):
        """Test updating server config."""
        mock_responses.add(
            responses.PUT,
            "http://localhost:3000/v1/admin/config",
            json={"max_vector_dim": 2048},
            status=200,
        )
        result = client.update_config({"max_vector_dim": 2048})
        assert result["max_vector_dim"] == 2048

    def test_get_quotas(self, client, mock_responses):
        """Test getting quotas via v1/admin path."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/admin/quotas",
            json={"global_max_vectors": 10000000},
            status=200,
        )
        result = client.get_quotas()
        assert result["global_max_vectors"] == 10000000

    def test_update_quotas(self, client, mock_responses):
        """Test updating quotas via v1/admin path."""
        mock_responses.add(
            responses.PUT,
            "http://localhost:3000/v1/admin/quotas",
            json={"global_max_vectors": 5000000},
            status=200,
        )
        result = client.update_quotas({"global_max_vectors": 5000000})
        assert result["global_max_vectors"] == 5000000


class TestKPIs:
    """Tests for KPI snapshot method."""

    def test_get_kpis(self, client, mock_responses):
        """Test getting KPI snapshot."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/kpis",
            json={
                "recall_latency_p50_ms": 2.5,
                "recall_latency_p99_ms": 15.0,
                "store_latency_p50_ms": 3.0,
                "api_error_rate_5xx_pct": 0.1,
                "active_agents_count": 12,
                "session_count_week": 500,
                "cross_agent_network_node_count": 150,
                "memory_retention_7d_pct": 95.5,
            },
            status=200,
        )
        result = client.get_kpis()
        assert result.recall_latency_p50_ms == 2.5
        assert result.recall_latency_p99_ms == 15.0
        assert result.active_agents_count == 12

    def test_get_kpis_error(self, client, mock_responses):
        """Test KPI endpoint with server error."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/kpis",
            json={"error": "metrics unavailable"},
            status=500,
        )
        with pytest.raises(ServerError):
            client.get_kpis()


class TestFulltextReindex:
    """Tests for CE-54 fulltext reindex."""

    def test_admin_fulltext_reindex(self, client, mock_responses):
        """Test triggering fulltext reindex."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/admin/fulltext/reindex",
            json={
                "total_indexed": 100,
                "total_skipped": 50,
                "namespaces": [
                    {"namespace": "ns-1", "indexed": 60, "skipped": 30},
                    {"namespace": "ns-2", "indexed": 40, "skipped": 20},
                ],
            },
            status=200,
        )
        result = client.admin_fulltext_reindex(namespace="ns-1")
        assert result.total_indexed == 100
        assert result.total_skipped == 50

    def test_admin_fulltext_reindex_all(self, client, mock_responses):
        """Test triggering fulltext reindex for all namespaces."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/admin/fulltext/reindex",
            json={"total_indexed": 200, "total_skipped": 100, "namespaces": []},
            status=200,
        )
        result = client.admin_fulltext_reindex()
        assert result.total_indexed == 200


class TestDrainReembed:
    """Tests for drain_reembed() — POST /admin/reembed/drain (v0.11.82+, DAK-6326)."""

    def test_drain_reembed_full_drain(self, client, mock_responses):
        """A full drain returns remaining=0 and parsed counters."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/admin/reembed/drain",
            json={
                "processed": 1280,
                "remaining": 0,
                "elapsed_ms": 4210,
                "cycles": 3,
                "timed_out": False,
            },
            status=200,
        )
        result = client.drain_reembed()
        assert result.processed == 1280
        assert result.remaining == 0
        assert result.cycles == 3
        assert result.timed_out is False
        assert len(mock_responses.calls) == 1
        # No params set -> empty/no body, never sends null fields
        body = mock_responses.calls[0].request.body
        assert body in (None, b"", b"{}", "{}")

    def test_drain_reembed_passes_params(self, client, mock_responses):
        """timeout_secs / batch_size / min_importance are forwarded in the body."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/admin/reembed/drain",
            json={
                "processed": 500,
                "remaining": 120,
                "elapsed_ms": 600000,
                "cycles": 50,
                "timed_out": True,
            },
            status=200,
        )
        result = client.drain_reembed(
            timeout_secs=600, batch_size=5000, min_importance=0.5
        )
        assert result.timed_out is True
        assert result.remaining == 120
        req_body = json.loads(mock_responses.calls[0].request.body)
        assert req_body["timeout_secs"] == 600
        assert req_body["batch_size"] == 5000
        assert req_body["min_importance"] == 0.5

    def test_drain_reembed_requires_admin_scope(self, client, mock_responses):
        """A 403 (missing Admin scope) surfaces as AuthorizationError."""
        from dakera import AuthorizationError

        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/admin/reembed/drain",
            json={"error": "admin scope required"},
            status=403,
        )
        with pytest.raises(AuthorizationError):
            client.drain_reembed()

    def test_drain_reembed_server_error(self, client, mock_responses):
        """A 500 surfaces as ServerError."""
        mock_responses.add(
            responses.POST,
            "http://localhost:3000/v1/admin/reembed/drain",
            json={"error": "internal failure"},
            status=500,
        )
        with pytest.raises(ServerError):
            client.drain_reembed()


class TestAsyncDrainReembed:
    """AsyncDakeraClient.drain_reembed() parity (DAK-6326)."""

    async def test_async_drain_reembed_parses_and_forwards_params(self):
        """Async variant forwards params and parses the DrainReembedResponse."""
        from unittest.mock import patch

        from dakera import AsyncDakeraClient

        client = AsyncDakeraClient("http://localhost:3000")
        captured: dict = {}

        async def fake_request(method, path, data=None, **kwargs):
            captured["method"] = method
            captured["path"] = path
            captured["data"] = data or {}
            return {
                "processed": 10,
                "remaining": 0,
                "elapsed_ms": 12,
                "cycles": 1,
                "timed_out": False,
            }

        with patch.object(client, "_request", side_effect=fake_request):
            result = await client.drain_reembed(timeout_secs=30, batch_size=100)

        assert captured["method"] == "POST"
        assert captured["path"] == "/v1/admin/reembed/drain"
        assert captured["data"]["timeout_secs"] == 30
        assert captured["data"]["batch_size"] == 100
        assert result.processed == 10
        assert result.remaining == 0
        assert result.timed_out is False


class TestAdminReembedStaticCount:
    """Tests for admin_reembed_static_count() (DAK-6781)."""

    def test_static_count_returns_count(self, client, mock_responses):
        """Happy-path: response is parsed into StaticCountResponse."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/admin/reembed/static-count",
            json={"static_count": 42},
            status=200,
        )
        result = client.admin_reembed_static_count()
        assert result.static_count == 42

    def test_static_count_zero_means_steady_state(self, client, mock_responses):
        """A static_count of 0 is returned cleanly."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/admin/reembed/static-count",
            json={"static_count": 0},
            status=200,
        )
        result = client.admin_reembed_static_count()
        assert result.static_count == 0

    def test_static_count_requires_admin_scope(self, client, mock_responses):
        """403 response surfaces as AuthorizationError."""
        from dakera import AuthorizationError

        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/admin/reembed/static-count",
            json={"error": "admin scope required"},
            status=403,
        )
        with pytest.raises(AuthorizationError):
            client.admin_reembed_static_count()

    def test_static_count_server_error(self, client, mock_responses):
        """500 response surfaces as ServerError."""
        mock_responses.add(
            responses.GET,
            "http://localhost:3000/v1/admin/reembed/static-count",
            json={"error": "internal failure"},
            status=500,
        )
        with pytest.raises(ServerError):
            client.admin_reembed_static_count()


class TestAsyncAdminReembedStaticCount:
    """AsyncDakeraClient.admin_reembed_static_count() parity (DAK-6781)."""

    async def test_async_static_count_parses_response(self):
        """Async variant issues GET and returns StaticCountResponse."""
        from unittest.mock import patch

        from dakera import AsyncDakeraClient

        client = AsyncDakeraClient("http://localhost:3000")
        captured: dict = {}

        async def fake_request(method, path, **kwargs):
            captured["method"] = method
            captured["path"] = path
            return {"static_count": 17}

        with patch.object(client, "_request", side_effect=fake_request):
            result = await client.admin_reembed_static_count()

        assert captured["method"] == "GET"
        assert captured["path"] == "/v1/admin/reembed/static-count"
        assert result.static_count == 17
