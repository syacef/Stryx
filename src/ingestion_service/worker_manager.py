import json
import logging
import threading
import time
from typing import Optional
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)

class WorkerManager:    
    WORKER_PREFIX = "worker:"
    WORKER_LIST_KEY = "workers:active"
    WORKER_STREAMS_PREFIX = "worker_streams:"
    ORPHANED_STREAMS_KEY = "streams:orphaned"

    def __init__(self, redis_client, stream_manager, max_streams_per_worker: int, worker_heartbeat_interval: int):
        self.redis = redis_client
        self.stream_manager = stream_manager
        self.running = False
        self.monitor_thread = None
        self.max_streams_per_worker = max_streams_per_worker
        self.worker_heartbeat_interval = worker_heartbeat_interval
        logger.info("WorkerManager initialized")
    
    def start(self):
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("WorkerManager started")
    
    def stop(self):
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("WorkerManager stopped")
    
    def _monitor_loop(self):
        while self.running:
            try:
                self._cleanup_stale_workers()
                self._recover_orphaned_streams()
                self._rebalance_if_needed()
                time.sleep(10)  # Check every 10 seconds
            except Exception as e:
                logger.error(f"Error in worker monitor: {e}")
                time.sleep(5)
    
    def register_worker(self, worker_id: str):
        worker_data = {
            "worker_id": worker_id,
            "status": "active",
            "created_at": datetime.now().isoformat(),
            "stream_count": 0,
            "last_heartbeat": datetime.now().isoformat(),
            "last_heartbeat_timestamp": time.time(),
            "active_stream_count": 0
        }
        
        worker_key = f"{self.WORKER_PREFIX}{worker_id}"
        self.redis.set(worker_key, json.dumps(worker_data))
        self.redis.expire(worker_key, self.worker_heartbeat_interval * 3)
        
        self.redis.sadd(self.WORKER_LIST_KEY, worker_id)
        
        worker_streams_key = f"{self.WORKER_STREAMS_PREFIX}{worker_id}"
        if not self.redis.exists(worker_streams_key):
            self.redis.delete(worker_streams_key)  # Ensure clean state
        
        logger.info(f"Worker {worker_id} registered")
    
    def unregister_worker(self, worker_id: str):
        logger.info(f"Unregistering worker {worker_id}")
        
        worker_streams_key = f"{self.WORKER_STREAMS_PREFIX}{worker_id}"
        stream_ids = self.redis.smembers(worker_streams_key)
        
        for stream_id in stream_ids:
            stream_id_str = stream_id.decode('utf-8') if isinstance(stream_id, bytes) else stream_id
            logger.info(f"Marking stream {stream_id_str} as orphaned")
            self.redis.sadd(self.ORPHANED_STREAMS_KEY, stream_id_str)
            self.stream_manager.update_worker_assignment(stream_id_str, None)
            self.stream_manager.update_stream_status(stream_id_str, "orphaned")
        
        # Delete worker metadata
        worker_key = f"{self.WORKER_PREFIX}{worker_id}"
        self.redis.delete(worker_key)
        self.redis.delete(worker_streams_key)
        
        # Remove from active workers
        self.redis.srem(self.WORKER_LIST_KEY, worker_id)
        
        logger.info(f"Worker {worker_id} unregistered, {len(stream_ids)} streams orphaned")
    
    def _cleanup_stale_workers(self):
        worker_ids = self.get_all_worker_ids()
        current_time = time.time()
        
        for worker_id in worker_ids:
            worker_key = f"{self.WORKER_PREFIX}{worker_id}"
            
            # Check if worker key still exists (TTL-based cleanup)
            if not self.redis.exists(worker_key):
                logger.warning(f"Worker {worker_id} key expired, cleaning up")
                self._cleanup_dead_worker(worker_id)
                continue
            
            worker_info = self.get_worker_info(worker_id)
            if not worker_info:
                continue
            
            last_heartbeat = worker_info.get("last_heartbeat_timestamp", 0)
            if current_time - last_heartbeat > self.worker_heartbeat_interval * 3:
                logger.warning(f"Worker {worker_id} appears stale (no heartbeat), cleaning up")
                self._cleanup_dead_worker(worker_id)
    
    def _cleanup_dead_worker(self, worker_id: str):
        worker_streams_key = f"{self.WORKER_STREAMS_PREFIX}{worker_id}"
        stream_ids = self.redis.smembers(worker_streams_key)
        
        for stream_id in stream_ids:
            stream_id_str = stream_id.decode('utf-8') if isinstance(stream_id, bytes) else stream_id
            logger.info(f"Orphaning stream {stream_id_str} from dead worker {worker_id}")
            self.redis.sadd(self.ORPHANED_STREAMS_KEY, stream_id_str)
            self.stream_manager.update_worker_assignment(stream_id_str, None)
            self.stream_manager.update_stream_status(stream_id_str, "orphaned")
        
        worker_key = f"{self.WORKER_PREFIX}{worker_id}"
        self.redis.delete(worker_key)
        self.redis.delete(worker_streams_key)
        
        self.redis.srem(self.WORKER_LIST_KEY, worker_id)
        
        logger.info(f"Dead worker {worker_id} cleaned up")
    
    def _recover_orphaned_streams(self):
        orphaned_streams = self.redis.smembers(self.ORPHANED_STREAMS_KEY)
        
        if not orphaned_streams:
            return
        
        logger.info(f"Found {len(orphaned_streams)} orphaned streams, recovering...")
        
        for stream_id in orphaned_streams:
            stream_id_str = stream_id.decode('utf-8') if isinstance(stream_id, bytes) else stream_id
            
            stream_info = self.stream_manager.get_stream_info(stream_id_str)
            if not stream_info:
                logger.warning(f"Orphaned stream {stream_id_str} no longer exists, removing")
                self.redis.srem(self.ORPHANED_STREAMS_KEY, stream_id_str)
                continue
            
            # Find available worker
            worker_id = self._find_available_worker()
            
            if not worker_id:
                logger.warning(f"No available workers to recover stream {stream_id_str}")
                continue
            
            # Assign stream to worker
            worker_streams_key = f"{self.WORKER_STREAMS_PREFIX}{worker_id}"
            self.redis.sadd(worker_streams_key, stream_id_str)
            
            self._update_worker_stream_count(worker_id)
            self.stream_manager.update_worker_assignment(stream_id_str, worker_id)
            self.stream_manager.update_stream_status(stream_id_str, "registered")
            
            self.redis.srem(self.ORPHANED_STREAMS_KEY, stream_id_str)
            
            logger.info(f"Stream {stream_id_str} recovered and assigned to worker {worker_id}")
    
    def _rebalance_if_needed(self):
        worker_ids = self.get_all_worker_ids()
        
        if len(worker_ids) < 2:
            return  # Need at least 2 workers to rebalance
        
        worker_loads = {}
        for worker_id in worker_ids:
            worker_streams_key = f"{self.WORKER_STREAMS_PREFIX}{worker_id}"
            worker_loads[worker_id] = self.redis.scard(worker_streams_key)
        
        if not worker_loads:
            return
        
        max_load = max(worker_loads.values())
        min_load = min(worker_loads.values())
        
        if max_load - min_load > 2:
            logger.info(f"Load imbalance detected (max: {max_load}, min: {min_load}), rebalancing...")
    
    def assign_worker_to_stream(self, stream_id: str) -> str:
        worker_id = self._find_available_worker()
        
        if not worker_id:
            logger.warning(f"No available workers for stream {stream_id}, marking as orphaned")
            self.redis.sadd(self.ORPHANED_STREAMS_KEY, stream_id)
            self.stream_manager.update_stream_status(stream_id, "orphaned")
            return "pending"
        
        worker_streams_key = f"{self.WORKER_STREAMS_PREFIX}{worker_id}"
        self.redis.sadd(worker_streams_key, stream_id)
        
        self._update_worker_stream_count(worker_id)
        self.stream_manager.update_worker_assignment(stream_id, worker_id)
        
        logger.info(f"Stream {stream_id} assigned to worker {worker_id}")
        return worker_id
    
    def unassign_worker_from_stream(self, stream_id: str):
        stream_info = self.stream_manager.get_stream_info(stream_id)
        if not stream_info or not stream_info.get("worker_id"):
            return
        
        worker_id = stream_info["worker_id"]
        
        worker_streams_key = f"{self.WORKER_STREAMS_PREFIX}{worker_id}"
        self.redis.srem(worker_streams_key, stream_id)
        
        self._update_worker_stream_count(worker_id)        
        self.stream_manager.update_worker_assignment(stream_id, None)
        
        self.redis.srem(self.ORPHANED_STREAMS_KEY, stream_id)
        
        logger.info(f"Stream {stream_id} unassigned from worker {worker_id}")
    
    def _find_available_worker(self) -> Optional[str]:
        worker_ids = self.get_all_worker_ids()
        
        if not worker_ids:
            return None
        
        # Find worker with least load
        min_load = self.max_streams_per_worker
        selected_worker = None
        
        for worker_id in worker_ids:
            worker_streams_key = f"{self.WORKER_STREAMS_PREFIX}{worker_id}"
            stream_count = self.redis.scard(worker_streams_key)

            if stream_count < self.max_streams_per_worker and stream_count < min_load:
                min_load = stream_count
                selected_worker = worker_id
        
        return selected_worker
    
    def _update_worker_stream_count(self, worker_id: str):
        worker_info = self.get_worker_info(worker_id)
        if not worker_info:
            return

        worker_streams_key = f"{self.WORKER_STREAMS_PREFIX}{worker_id}"
        stream_count = self.redis.scard(worker_streams_key)
        
        worker_info["stream_count"] = stream_count
        worker_info["updated_at"] = datetime.now().isoformat()
        
        worker_key = f"{self.WORKER_PREFIX}{worker_id}"
        self.redis.set(worker_key, json.dumps(worker_info))
        self.redis.expire(worker_key, self.worker_heartbeat_interval * 3)
    
    def get_worker_info(self, worker_id: str) -> Optional[dict]:
        worker_key = f"{self.WORKER_PREFIX}{worker_id}"
        worker_data = self.redis.get(worker_key)
        
        if not worker_data:
            return None
        
        worker_info = json.loads(worker_data)
        
        # Add assigned streams
        worker_streams_key = f"{self.WORKER_STREAMS_PREFIX}{worker_id}"
        stream_ids = self.redis.smembers(worker_streams_key)
        worker_info["assigned_streams"] = [
            s.decode('utf-8') if isinstance(s, bytes) else s for s in stream_ids
        ]
        
        return worker_info
    
    def get_all_worker_ids(self) -> list[str]:
        worker_ids = self.redis.smembers(self.WORKER_LIST_KEY)
        return [
            w.decode('utf-8') if isinstance(w, bytes) else w for w in worker_ids
        ]
    
    def get_active_worker_count(self) -> int:
        return self.redis.scard(self.WORKER_LIST_KEY)
    
    def get_worker_status(self) -> list[dict]:
        worker_ids = self.get_all_worker_ids()
        status_list = []
        
        for worker_id in worker_ids:
            worker_info = self.get_worker_info(worker_id)
            if worker_info:
                status_list.append({
                    "worker_id": worker_id,
                    "status": worker_info.get("status", "unknown"),
                    "assigned_streams": worker_info.get("assigned_streams", []),
                    "stream_count": len(worker_info.get("assigned_streams", [])),
                    "last_heartbeat": worker_info.get("last_heartbeat")
                })
        
        return status_list
