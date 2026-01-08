def test_import_package():
    import cfllm  # noqa: F401


def test_import_key_modules():
    from cfllm import config  # noqa: F401
    from cfllm import env  # noqa: F401
