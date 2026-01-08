"""
Async operations support for MARIE agents.

Provides async wrappers and parallelization utilities.
"""

import asyncio
from typing import List, Dict, Any, Callable, Optional, Coroutine, TypeVar
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class AsyncRunner:
    """
    Async operation runner with thread pool support.
    
    Allows running sync functions in async context and vice versa.
    """
    
    def __init__(self, max_workers: int = 10):
        """
        Initialize async runner.
        
        Args:
            max_workers: Maximum thread pool workers
        """
        self.max_workers = max_workers
        self._executor: Optional[ThreadPoolExecutor] = None
        logger.info(f"AsyncRunner initialized with {max_workers} workers")
    
    def __enter__(self):
        """Context manager entry."""
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
    
    async def run_in_executor(
        self,
        func: Callable[..., T],
        *args,
        **kwargs
    ) -> T:
        """
        Run sync function in thread pool executor.
        
        Args:
            func: Sync function to run
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
        """
        if not self._executor:
            self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        loop = asyncio.get_event_loop()
        
        # Create partial function with kwargs
        if kwargs:
            from functools import partial
            func = partial(func, **kwargs)
        
        return await loop.run_in_executor(self._executor, func, *args)
    
    async def gather_with_concurrency(
        self,
        n: int,
        *coros: Coroutine
    ) -> List[Any]:
        """
        Run coroutines with limited concurrency.
        
        Args:
            n: Max concurrent operations
            *coros: Coroutines to run
            
        Returns:
            List of results
        """
        semaphore = asyncio.Semaphore(n)
        
        async def run_with_semaphore(coro):
            async with semaphore:
                return await coro
        
        return await asyncio.gather(
            *[run_with_semaphore(c) for c in coros],
            return_exceptions=False
        )


async def run_parallel(
    tasks: List[Callable[[], T]],
    max_concurrent: Optional[int] = None
) -> List[T]:
    """
    Run sync tasks in parallel using thread pool.
    
    Args:
        tasks: List of sync functions to run
        max_concurrent: Max concurrent tasks (None = unlimited)
        
    Returns:
        List of results in same order as tasks
        
    Example:
        ```python
        results = await run_parallel([
            lambda: agent1.run(),
            lambda: agent2.run(),
            lambda: agent3.run()
        ], max_concurrent=2)
        ```
    """
    max_workers = max_concurrent or len(tasks)
    
    with AsyncRunner(max_workers=max_workers) as runner:
        coros = [runner.run_in_executor(task) for task in tasks]
        
        if max_concurrent:
            results = await runner.gather_with_concurrency(max_concurrent, *coros)
        else:
            results = await asyncio.gather(*coros, return_exceptions=False)
    
    logger.info(f"Completed {len(tasks)} parallel tasks")
    return results


async def run_parallel_dict(
    tasks: Dict[str, Callable[[], T]],
    max_concurrent: Optional[int] = None
) -> Dict[str, T]:
    """
    Run sync tasks in parallel and return dict of results.
    
    Args:
        tasks: Dict of name -> sync function
        max_concurrent: Max concurrent tasks
        
    Returns:
        Dict of name -> result
        
    Example:
        ```python
        results = await run_parallel_dict({
            'entity': lambda: entity_agent.run(),
            'retrieval': lambda: retrieval_agent.run()
        })
        print(results['entity'])
        ```
    """
    names = list(tasks.keys())
    funcs = list(tasks.values())
    
    results = await run_parallel(funcs, max_concurrent=max_concurrent)
    
    return dict(zip(names, results))


async def run_with_timeout(
    coro: Coroutine[Any, Any, T],
    timeout: float
) -> T:
    """
    Run coroutine with timeout.
    
    Args:
        coro: Coroutine to run
        timeout: Timeout in seconds
        
    Returns:
        Result
        
    Raises:
        asyncio.TimeoutError: If timeout exceeded
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.error(f"Operation timed out after {timeout}s")
        raise


async def run_with_retry(
    coro_factory: Callable[[], Coroutine[Any, Any, T]],
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0
) -> T:
    """
    Run coroutine with retry logic.
    
    Args:
        coro_factory: Factory function that creates coroutine
        max_retries: Maximum retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Backoff multiplier
        
    Returns:
        Result
        
    Raises:
        Exception: Last exception if all retries fail
        
    Example:
        ```python
        result = await run_with_retry(
            lambda: call_api(),
            max_retries=3,
            delay=1.0
        )
        ```
    """
    last_exception = None
    current_delay = delay
    
    for attempt in range(max_retries + 1):
        try:
            coro = coro_factory()
            return await coro
        except Exception as e:
            last_exception = e
            
            if attempt < max_retries:
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                    f"Retrying in {current_delay}s..."
                )
                await asyncio.sleep(current_delay)
                current_delay *= backoff
            else:
                logger.error(f"All {max_retries + 1} attempts failed")
    
    raise last_exception


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """
    Run coroutine in sync context (creates event loop if needed).
    
    Args:
        coro: Coroutine to run
        
    Returns:
        Result
        
    Note:
        Use with caution - creates new event loop if none exists.
        Prefer using async/await in async contexts.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Already in async context - create new loop in thread
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(coro)
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop - create new one
        return asyncio.run(coro)


class BatchProcessor:
    """
    Process items in batches with async support.
    
    Useful for batching API calls, DB queries, etc.
    """
    
    def __init__(
        self,
        batch_size: int = 10,
        max_concurrent_batches: int = 3
    ):
        """
        Initialize batch processor.
        
        Args:
            batch_size: Items per batch
            max_concurrent_batches: Max concurrent batch processing
        """
        self.batch_size = batch_size
        self.max_concurrent_batches = max_concurrent_batches
    
    async def process_async(
        self,
        items: List[Any],
        process_fn: Callable[[List[Any]], Coroutine[Any, Any, List[T]]]
    ) -> List[T]:
        """
        Process items in batches asynchronously.
        
        Args:
            items: Items to process
            process_fn: Async function that processes a batch
            
        Returns:
            All results flattened
        """
        # Split into batches
        batches = [
            items[i:i + self.batch_size]
            for i in range(0, len(items), self.batch_size)
        ]
        
        logger.info(f"Processing {len(items)} items in {len(batches)} batches")
        
        # Process batches with concurrency limit
        semaphore = asyncio.Semaphore(self.max_concurrent_batches)
        
        async def process_batch(batch):
            async with semaphore:
                return await process_fn(batch)
        
        batch_results = await asyncio.gather(
            *[process_batch(batch) for batch in batches],
            return_exceptions=False
        )
        
        # Flatten results
        results = []
        for batch_result in batch_results:
            if isinstance(batch_result, list):
                results.extend(batch_result)
            else:
                results.append(batch_result)
        
        logger.info(f"Completed processing {len(results)} results")
        return results
    
    def process_sync(
        self,
        items: List[Any],
        process_fn: Callable[[List[Any]], List[T]]
    ) -> List[T]:
        """
        Process items in batches synchronously with parallelization.
        
        Args:
            items: Items to process
            process_fn: Sync function that processes a batch
            
        Returns:
            All results flattened
        """
        # Split into batches
        batches = [
            items[i:i + self.batch_size]
            for i in range(0, len(items), self.batch_size)
        ]
        
        logger.info(f"Processing {len(items)} items in {len(batches)} batches")
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=self.max_concurrent_batches) as executor:
            batch_results = list(executor.map(process_fn, batches))
        
        # Flatten results
        results = []
        for batch_result in batch_results:
            if isinstance(batch_result, list):
                results.extend(batch_result)
            else:
                results.append(batch_result)
        
        logger.info(f"Completed processing {len(results)} results")
        return results


# Utility decorators
def async_cached(cache, ttl: Optional[int] = None):
    """
    Decorator for async functions with caching.
    
    Args:
        cache: Cache instance (must have async get/set)
        ttl: TTL in seconds
        
    Example:
        ```python
        @async_cached(get_redis_schema_cache(), ttl=600)
        async def get_schema(index: str):
            # expensive operation
            return schema
        ```
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Create cache key from function name and args
            import hashlib
            import json
            
            key_data = {
                'func': func.__name__,
                'args': str(args),
                'kwargs': str(sorted(kwargs.items()))
            }
            key = hashlib.md5(json.dumps(key_data).encode()).hexdigest()
            
            # Try cache
            cached = cache.get(key)
            if cached is not None:
                return cached
            
            # Call function
            result = await func(*args, **kwargs)
            
            # Cache result
            cache.set(key, result, ttl)
            
            return result
        
        return wrapper
    return decorator
