from app import log_error, tracer

def test_log_error():
    '''Vérifie que la fonction log_error ne génère pas d'erreurs imprévues
    et s'exécute correctement.'''
    try:
        log_error("TestSpan", "An error occurred")
    except Exception as e:
        assert False, f"Log error failed: {e}"

def test_tracer():
    '''Vérifie que le traceur crée un span avec le nom correct'''
    with tracer.start_as_current_span("TestSpan") as span:
        span.set_attribute("test_attribute", "test_value")
    assert span.name == "TestSpan"
