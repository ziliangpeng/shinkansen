from shinkansen import __version__


def test_version_format():
    assert isinstance(__version__, str)
    assert __version__.count(".") >= 2
