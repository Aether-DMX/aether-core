#!/usr/bin/env python3
"""
Test script for pixel_mapper with your specific configuration:
- Universe: 4
- 5 RGBW fixtures at channels 1, 5, 9, 13, 17
- 4-channel spacing (contiguous RGBW)
"""

from pixel_mapper import (
    PixelArrayController, Pixel, PixelArrayConfig,
    OperationMode, EffectType, create_pixel_controller
)
import time

def main():
    print('='*60)
    print('TESTING: Universe 4, 5 RGBW fixtures at 1, 5, 9, 13, 17')
    print('='*60)

    # Create controller with your exact configuration
    ctrl = create_pixel_controller(
        fixture_count=5,
        universe=4,
        start_channel=1,
        channel_spacing=4
    )

    # Print fixture map
    ctrl.print_fixture_map()

    # Validate
    is_valid, errors = ctrl.validate()
    print(f'Validation: {"PASSED" if is_valid else "FAILED"}')
    if errors:
        for e in errors:
            print(f'  ERROR: {e}')

    # Test grouped mode - all red
    print('\n' + '='*60)
    print('TEST 1: Grouped Mode - All fixtures RED')
    print('='*60)
    ctrl.set_grouped_mode()
    ctrl.set_all_rgbw(255, 0, 0, 0)
    ctrl.populate_dmx_buffer()
    channels = ctrl.get_dmx_channels()
    print(f'DMX Channels sent: {channels}')

    # Test pixel array mode - gradient
    print('\n' + '='*60)
    print('TEST 2: Pixel Array Mode - Red gradient')
    print('='*60)
    ctrl.set_pixel_array_mode()
    for i in range(5):
        brightness = int(255 * (i / 4))
        ctrl.set_pixel_rgbw(i, brightness, 0, 0, 0)
        print(f'  Fixture {i}: R={brightness}')
    ctrl.populate_dmx_buffer()
    channels = ctrl.get_dmx_channels()
    print(f'DMX Channels sent: {channels}')

    # Test wave effect (single frame)
    print('\n' + '='*60)
    print('TEST 3: Wave Effect (single frame at t=0)')
    print('='*60)
    ctrl.set_effect(EffectType.WAVE, color=Pixel(0, 255, 0, 0), speed=1.0)
    ctrl._start_time = time.monotonic()
    channels = ctrl.render_single_frame()
    print(f'DMX Channels sent: {channels}')

    # Test chase effect
    print('\n' + '='*60)
    print('TEST 4: Chase Effect (single frame)')
    print('='*60)
    ctrl.set_effect(EffectType.CHASE, color=Pixel(0, 0, 255, 0), speed=1.0, tail_length=2)
    channels = ctrl.render_single_frame()
    print(f'DMX Channels sent: {channels}')

    # Print full status
    print('\n' + '='*60)
    print('CONTROLLER STATUS')
    print('='*60)
    status = ctrl.get_status()
    print(f'Universe: {status["universe"]}')
    print(f'Fixture Count: {status["fixture_count"]}')
    print(f'Start Channel: {status["start_channel"]}')
    print(f'Channel Spacing: {status["channel_spacing"]}')
    print(f'Mode: {status["mode"]}')
    print(f'Effect: {status["effect_type"]}')
    print(f'FPS Target: {status["target_fps"]}')
    print()
    print('Pixels:')
    for i, p in enumerate(status['pixels']):
        print(f'  [{i}] R={p["r"]:3d} G={p["g"]:3d} B={p["b"]:3d} W={p["w"]:3d}')

    print('\n' + '='*60)
    print('ALL TESTS COMPLETE')
    print('='*60)

if __name__ == '__main__':
    main()
