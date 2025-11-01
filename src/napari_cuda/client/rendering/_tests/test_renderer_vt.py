from napari_cuda.client.rendering.renderer import (
    GL,
    GLRenderer,
    VTReleaseQueue,
)


class DummyVT:
    def __init__(self):
        self.released = []

    def gl_release_tex(self, tex_cap):
        self.released.append(tex_cap)

    def gl_cache_init_for_current_context(self):
        return object()


def test_vt_release_queue_defers_until_signaled(monkeypatch):
    queue = VTReleaseQueue()
    vt = DummyVT()
    sync_handle = object()

    calls = {'wait': 0, 'deleted': 0}

    def fake_fence(_flag1, _flag2):
        return sync_handle

    def fake_wait(sync, *_args):
        assert sync is sync_handle
        calls['wait'] += 1
        # First drain should observe timeout, second should succeed
        return GL.GL_TIMEOUT_EXPIRED if calls['wait'] == 1 else GL.GL_ALREADY_SIGNALED

    def fake_delete(sync):
        assert sync is sync_handle
        calls['deleted'] += 1

    monkeypatch.setattr(GL, 'glFenceSync', fake_fence)
    monkeypatch.setattr(GL, 'glClientWaitSync', fake_wait)
    monkeypatch.setattr(GL, 'glDeleteSync', fake_delete)

    queued = queue.enqueue('tex_cap')
    assert queued
    # First drain: fence not ready, no release
    queue.drain(vt)
    assert vt.released == []
    # Second drain: fence signaled, release happens
    queue.drain(vt)
    assert vt.released == ['tex_cap']
    assert calls['deleted'] == 1


def test_context_change_resets_queue_and_resources(monkeypatch):
    renderer = object.__new__(GLRenderer)
    renderer._vt_release_queue = VTReleaseQueue()
    renderer._vt = DummyVT()
    renderer._gl_prog_rect = 101
    renderer._gl_prog_2d = 202
    renderer._gl_vbo = 303
    renderer._gl_pos_loc_rect = 1
    renderer._gl_tex_loc_rect = 2
    renderer._gl_u_tex_size_loc = 3
    renderer._gl_pos_loc_2d = 4
    renderer._gl_tex_loc_2d = 5
    renderer._vt_cache = 'cache'
    renderer._vt_first_draw_logged = True
    renderer._gl_frame_counter = 42

    deletes = []

    monkeypatch.setattr(GL, 'glDeleteProgram', lambda handle: deletes.append(('prog', handle)))
    monkeypatch.setattr(GL, 'glDeleteBuffers', lambda _count, handles: deletes.append(('vbo', handles[0])))

    # Pretend the release queue still owns a texture
    renderer._vt_release_queue._entries.append((None, 'tex_cap'))

    renderer._on_context_changed()

    assert renderer._vt_cache is None
    assert renderer._gl_frame_counter == 0
    assert renderer._gl_prog_rect is None
    assert renderer._gl_prog_2d is None
    assert renderer._gl_vbo is None
    assert renderer._vt_first_draw_logged is False
    assert renderer._vt.released == ['tex_cap']
    assert ('prog', 101) in deletes and ('prog', 202) in deletes and ('vbo', 303) in deletes
