@pytest.mark.parametrize("hook_mode", ["preload", "torch"])
def test_simple(hook_mode):
    configure_tms_and_run_in_subprocess(_test_simple_inner, hook_mode=hook_mode)
