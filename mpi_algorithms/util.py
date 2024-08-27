import queue
import threading
import time
from typing import Any, Callable, List, Optional, TypeGuard, TypeVar

T = TypeVar("T")


def no_nones_list(lst: List[Optional[T]]) -> TypeGuard[List[T]]:
    """
    Returns True if the given list has no Nones.
    """
    return all(x is not None for x in lst)


def run_threads(functions: List[Callable[..., Any]], args: List[Any]) -> List[Any]:
    """
    Runs the given functions in separate threads until done, propagating any exceptions.
    """
    exc_queue: queue.Queue[Exception] = queue.Queue()
    threads: List[threading.Thread] = []
    results = [None] * len(functions)
    for i, (f, a) in enumerate(zip(functions, args)):

        def wrapped_f(index: int, args: Any) -> None:
            try:
                results[index] = f(*args)
            except Exception as e:
                exc_queue.put(e)

        thread = threading.Thread(target=wrapped_f, args=(i, a))
        thread.start()
        threads.append(thread)
    while True:
        if not exc_queue.empty():
            raise exc_queue.get()
        if all(not t.is_alive() for t in threads):
            break
        time.sleep(0.1)
    return results
