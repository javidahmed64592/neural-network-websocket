from example.main import example_function


def test_example_function() -> None:
    result = example_function()
    assert result == "This is an example function."
