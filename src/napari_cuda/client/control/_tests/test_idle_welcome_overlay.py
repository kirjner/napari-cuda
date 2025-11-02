from napari_cuda.client.control.mirrors.napari_layer_mirror import NapariLayerMirror


def test_should_show_welcome_overlay_logic() -> None:
    assert NapariLayerMirror._should_show_welcome({"status": "idle"}, layer_count=5)
    assert not NapariLayerMirror._should_show_welcome({"status": "ready"}, layer_count=5)
    assert NapariLayerMirror._should_show_welcome({}, layer_count=0)
    assert not NapariLayerMirror._should_show_welcome(None, layer_count=1)
