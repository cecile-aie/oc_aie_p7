from app import log_error, tracer

def test_log_error():
    try:
        log_error("TestSpan", "An error occurred")
    except Exception as e:
        assert False, f"Log error failed: {e}"

def test_tracer():
    with tracer.start_as_current_span("TestSpan") as span:
        span.set_attribute("test_attribute", "test_value")
    assert span.name == "TestSpan"
