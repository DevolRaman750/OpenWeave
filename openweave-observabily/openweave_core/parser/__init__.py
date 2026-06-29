__all__ = ["parse_langfuse_trace", "parse_langfuse_trace_full"]


def __getattr__(name):
    if name == "parse_langfuse_trace":
        from .trace_fetcher import parse_langfuse_trace

        return parse_langfuse_trace
    if name == "parse_langfuse_trace_full":
        from .trace_fetcher import parse_langfuse_trace_full

        return parse_langfuse_trace_full
    raise AttributeError(name)
