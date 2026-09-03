from drakonix_lora.gui import build_app


def test_build_app() -> None:
    app = build_app()
    assert app is not None
