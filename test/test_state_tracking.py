#!/usr/bin/env python3
"""
Test for state tracking functionality in TorchMemorySaver.
Verifies that pause/resume operations respect allocation states.
"""

import torch
import torch_memory_saver
import pytest


def test_state_tracking_basic():
    """Test basic state tracking - pause and resume operations"""
    tms = torch_memory_saver.TorchMemorySaver()

    with tms.region(tag="test_tag"):
        # Allocate some memory
        tensor1 = torch.randn(1000, 1000, device='cuda')
        tensor2 = torch.randn(500, 500, device='cuda')

        # First pause should work
        tms.pause("test_tag")

        # Second pause on same tag should throw an error (already paused)
        with pytest.raises(RuntimeError, match="Cannot pause allocation that is already paused"):
            tms.pause("test_tag")

        # Resume should work
        tms.resume("test_tag")

        # Second resume on same tag should throw an error (already active)
        with pytest.raises(RuntimeError, match="Cannot resume allocation that is already active"):
            tms.resume("test_tag")

        # Another pause should work
        tms.pause("test_tag")

        # Another resume should work
        tms.resume("test_tag")


def test_state_tracking_multiple_tags():
    """Test state tracking with multiple tags"""
    tms = torch_memory_saver.TorchMemorySaver()

    with tms.region(tag="tag1"):
        tensor1 = torch.randn(100, 100, device='cuda')

    with tms.region():
        tensor2 = torch.randn(200, 200, device='cuda')

    tms.pause("tag1")

    # Pause tag1 again (should throw error)
    with pytest.raises(RuntimeError, match="Cannot pause allocation that is already paused"):
        tms.pause("tag1")

    tms.resume()

    # Resume default tag (should throw error)
    with pytest.raises(RuntimeError, match="Cannot resume allocation that is already active"):
        tms.resume()



def test_error_message_contains_tag():
    """Test that error messages contain the relevant tag information"""
    tms = torch_memory_saver.TorchMemorySaver()

    with tms.region(tag="test_tag"):
        tensor = torch.randn(100, 100, device='cuda')

    with pytest.raises(RuntimeError) as exc_info:
        tms.resume("test_tag")

    error_msg = str(exc_info.value)
    assert "test_tag" in error_msg
    assert "already active" in error_msg


if __name__ == "__main__":
    test_state_tracking_basic()
    test_state_tracking_multiple_tags()
    test_error_message_contains_tag()
    print("All state tracking tests passed!")