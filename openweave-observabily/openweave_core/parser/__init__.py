__all__ = ["parse_langfuse_trace"]


def __getattr__(name):
    if name == "parse_langfuse_trace":
        from .trace_fetcher import parse_langfuse_trace

        return parse_langfuse_trace
    raise AttributeError(name)
