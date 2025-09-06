"""
Performance monitoring and metrics collection for the RAG system.
Tracks timing, memory usage, throughput, and system performance metrics.
"""

import time
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import os
from datetime import datetime, timedelta
import logging
import functools

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Individual performance metric."""
    operation: str
    start_time: float
    end_time: float
    duration: float
    memory_start: Optional[float] = None
    memory_end: Optional[float] = None
    memory_delta: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemMetrics:
    """System-level performance metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_available: float
    disk_usage_percent: float
    network_io: Dict[str, float]


class PerformanceMonitor:
    """Comprehensive performance monitoring for the RAG system."""
    
    def __init__(self, max_history: int = 1000, enable_system_monitoring: bool = True):
        self.max_history = max_history
        self.enable_system_monitoring = enable_system_monitoring
        
        # Performance metrics storage
        self.metrics: deque = deque(maxlen=max_history)
        self.operation_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'count': 0,
            'total_time': 0.0,
            'avg_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'success_count': 0,
            'error_count': 0,
            'memory_usage': []
        })
        
        # System monitoring
        self.system_metrics: deque = deque(maxlen=max_history)
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_monitoring = False
        
        # Performance thresholds
        self.slow_operation_threshold = 5.0  # seconds
        self.memory_warning_threshold = 80.0  # percent
        self.cpu_warning_threshold = 90.0  # percent
        
        if enable_system_monitoring:
            self.start_system_monitoring()
    
    def start_system_monitoring(self):
        """Start background system monitoring."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            return
        
        self.stop_monitoring = False
        self.monitoring_thread = threading.Thread(target=self._monitor_system, daemon=True)
        self.monitoring_thread.start()
        logger.info("System monitoring started")
    
    def stop_system_monitoring(self):
        """Stop background system monitoring."""
        self.stop_monitoring = True
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        logger.info("System monitoring stopped")
    
    def _monitor_system(self):
        """Background system monitoring loop."""
        while not self.stop_monitoring:
            try:
                metrics = self._collect_system_metrics()
                self.system_metrics.append(metrics)
                
                # Check for warnings
                self._check_system_warnings(metrics)
                
                time.sleep(5)  # Collect metrics every 5 seconds
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                time.sleep(10)  # Wait longer on error
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network I/O
            net_io = psutil.net_io_counters()
            network_io = {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv
            }
            
            return SystemMetrics(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_available=memory.available / (1024**3),  # GB
                disk_usage_percent=disk.percent,
                network_io=network_io
            )
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return SystemMetrics(
                timestamp=time.time(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_available=0.0,
                disk_usage_percent=0.0,
                network_io={}
            )
    
    def _check_system_warnings(self, metrics: SystemMetrics):
        """Check system metrics for warning conditions."""
        if metrics.cpu_percent > self.cpu_warning_threshold:
            logger.warning(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        if metrics.memory_percent > self.memory_warning_threshold:
            logger.warning(f"High memory usage: {metrics.memory_percent:.1f}%")
        
        if metrics.disk_usage_percent > 90:
            logger.warning(f"High disk usage: {metrics.disk_usage_percent:.1f}%")
    
    def start_operation(self, operation: str, metadata: Dict[str, Any] = None) -> str:
        """Start timing an operation and return operation ID."""
        operation_id = f"{operation}_{int(time.time() * 1000000)}"
        
        metric = PerformanceMetric(
            operation=operation,
            start_time=time.time(),
            end_time=0.0,
            duration=0.0,
            memory_start=self._get_memory_usage(),
            metadata=metadata or {}
        )
        
        self.metrics.append(metric)
        return operation_id
    
    def end_operation(self, operation_id: str, success: bool = True, 
                     error_message: str = None, metadata: Dict[str, Any] = None):
        """End timing an operation."""
        # Find the metric by operation ID
        for metric in reversed(self.metrics):
            if metric.operation in operation_id:
                metric.end_time = time.time()
                metric.duration = metric.end_time - metric.start_time
                metric.memory_end = self._get_memory_usage()
                metric.success = success
                metric.error_message = error_message
                
                if metadata:
                    metric.metadata.update(metadata)
                
                # Update operation statistics
                self._update_operation_stats(metric)
                
                # Check for slow operations
                if metric.duration > self.slow_operation_threshold:
                    logger.warning(f"Slow operation detected: {metric.operation} took {metric.duration:.2f}s")
                
                break
    
    def _get_memory_usage(self) -> Optional[float]:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return None
    
    def _update_operation_stats(self, metric: PerformanceMetric):
        """Update operation statistics."""
        stats = self.operation_stats[metric.operation]
        
        stats['count'] += 1
        stats['total_time'] += metric.duration
        stats['avg_time'] = stats['total_time'] / stats['count']
        stats['min_time'] = min(stats['min_time'], metric.duration)
        stats['max_time'] = max(stats['max_time'], metric.duration)
        
        if metric.success:
            stats['success_count'] += 1
        else:
            stats['error_count'] += 1
        
        if metric.memory_delta is not None:
            stats['memory_usage'].append(metric.memory_delta)
            # Keep only recent memory usage data
            if len(stats['memory_usage']) > 100:
                stats['memory_usage'] = stats['memory_usage'][-100:]
    
    def get_operation_stats(self, operation: str = None) -> Dict[str, Any]:
        """Get statistics for specific operation or all operations."""
        if operation:
            return self.operation_stats.get(operation, {})
        return dict(self.operation_stats)
    
    def get_recent_metrics(self, minutes: int = 60) -> List[PerformanceMetric]:
        """Get metrics from the last N minutes."""
        cutoff_time = time.time() - (minutes * 60)
        return [m for m in self.metrics if m.start_time > cutoff_time]
    
    def get_system_metrics_summary(self, minutes: int = 60) -> Dict[str, Any]:
        """Get system metrics summary for the last N minutes."""
        cutoff_time = time.time() - (minutes * 60)
        recent_metrics = [m for m in self.system_metrics if m.timestamp > cutoff_time]
        
        if not recent_metrics:
            return {}
        
        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_percent for m in recent_metrics]
        
        return {
            'cpu': {
                'avg': sum(cpu_values) / len(cpu_values),
                'max': max(cpu_values),
                'min': min(cpu_values)
            },
            'memory': {
                'avg': sum(memory_values) / len(memory_values),
                'max': max(memory_values),
                'min': min(memory_values)
            },
            'sample_count': len(recent_metrics),
            'time_range_minutes': minutes
        }
    
    def export_metrics(self, filepath: str, include_system: bool = True):
        """Export metrics to JSON file."""
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'operation_stats': dict(self.operation_stats),
                'recent_metrics': [
                    {
                        'operation': m.operation,
                        'start_time': datetime.fromtimestamp(m.start_time).isoformat(),
                        'duration': m.duration,
                        'success': m.success,
                        'memory_delta': m.memory_delta,
                        'metadata': m.metadata
                    }
                    for m in list(self.metrics)[-100:]  # Last 100 metrics
                ]
            }
            
            if include_system and self.system_metrics:
                export_data['system_metrics'] = [
                    {
                        'timestamp': datetime.fromtimestamp(m.timestamp).isoformat(),
                        'cpu_percent': m.cpu_percent,
                        'memory_percent': m.memory_percent,
                        'memory_available_gb': m.memory_available
                    }
                    for m in list(self.system_metrics)[-100:]  # Last 100 system metrics
                ]
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Metrics exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
    
    def clear_metrics(self):
        """Clear all stored metrics."""
        self.metrics.clear()
        self.system_metrics.clear()
        self.operation_stats.clear()
        logger.info("All performance metrics cleared")
    
    def get_performance_summary(self) -> str:
        """Get a human-readable performance summary."""
        if not self.operation_stats:
            return "No performance data available"
        
        summary_parts = ["Performance Summary:"]
        summary_parts.append("=" * 50)
        
        for operation, stats in self.operation_stats.items():
            if stats['count'] == 0:
                continue
            
            success_rate = (stats['success_count'] / stats['count']) * 100
            summary_parts.append(f"\n{operation}:")
            summary_parts.append(f"  Total calls: {stats['count']}")
            summary_parts.append(f"  Success rate: {success_rate:.1f}%")
            summary_parts.append(f"  Avg time: {stats['avg_time']:.4f}s")
            summary_parts.append(f"  Min time: {stats['min_time']:.4f}s")
            summary_parts.append(f"  Max time: {stats['max_time']:.4f}s")
        
        # Add system metrics if available
        if self.system_metrics:
            sys_summary = self.get_system_metrics_summary(minutes=5)
            if sys_summary:
                summary_parts.append(f"\nSystem Metrics (last 5 minutes):")
                summary_parts.append(f"  CPU: avg={sys_summary['cpu']['avg']:.1f}%, max={sys_summary['cpu']['max']:.1f}%")
                summary_parts.append(f"  Memory: avg={sys_summary['memory']['avg']:.1f}%, max={sys_summary['memory']['max']:.1f}%")
        
        return "\n".join(summary_parts)


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


# Convenience functions for easy usage
def monitor_operation(operation_name: str = None):
    """Decorator to monitor function performance."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            operation_id = performance_monitor.start_operation(op_name)
            
            try:
                result = func(*args, **kwargs)
                performance_monitor.end_operation(operation_id, success=True)
                return result
            except Exception as e:
                performance_monitor.end_operation(operation_id, success=False, error_message=str(e))
                raise e
        
        return wrapper
    return decorator


def get_performance_stats(operation: str = None) -> Dict[str, Any]:
    """Get performance statistics for an operation or all operations."""
    return performance_monitor.get_operation_stats(operation)


def export_performance_data(filepath: str):
    """Export performance data to file."""
    performance_monitor.export_metrics(filepath)
