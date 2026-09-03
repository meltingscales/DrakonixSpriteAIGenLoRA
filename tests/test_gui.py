from drakonix_lora.gui import _render_queue_status, build_app


def test_build_app() -> None:
    app = build_app()
    assert app is not None


def test_render_queue_status_marks_done_active_and_queued() -> None:
    prompts = ["a knight", "a dragon", "a frog"]
    text = _render_queue_status(prompts, current_idx=1, current_desc="step 3/10")

    lines = text.splitlines()
    assert "done" in lines[0] and "a knight" in lines[0]
    assert "active" in lines[1] and "a dragon" in lines[1] and "step 3/10" in lines[1]
    assert "queued" in lines[2] and "a frog" in lines[2]


def test_render_queue_status_truncates_long_prompts() -> None:
    long_prompt = "pixelart style, " + "a very long description " * 5
    text = _render_queue_status([long_prompt], current_idx=0, current_desc="")
    assert "..." in text
