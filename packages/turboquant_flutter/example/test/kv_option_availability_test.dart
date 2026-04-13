import 'package:flutter_test/flutter_test.dart';
import 'package:turboquant_flutter/src/api/turboquant_api.dart';
import 'package:turboquant_flutter_example/main.dart';

void main() {
  test('TurboQuant option prefers GPU-KV validation when available', () {
    final option = KvOptionAvailability(
      cacheType: 'turbo4',
      cpuKvResult: TQValidationResult(
        success: true,
        gpuLayersEnabled: true,
        offloadKv: false,
        cpuKvFallback: true,
        flashAttentionAuto: true,
        flashAttentionRequired: true,
        nGpuLayers: 99,
      ),
      gpuKvResult: TQValidationResult(
        success: true,
        gpuLayersEnabled: true,
        offloadKv: true,
        cpuKvFallback: false,
        flashAttentionAuto: true,
        flashAttentionRequired: true,
        nGpuLayers: 99,
      ),
    );

    expect(option.isAvailable, isTrue);
    expect(option.preferredResult?.pathLabel, 'GPU-KV');
    expect(option.dropdownLabel, 'turbo4 (GPU-KV validated)');
  });

  test('TurboQuant option exposes CPU fallback when GPU-KV fails', () {
    final option = KvOptionAvailability(
      cacheType: 'turbo3',
      cpuKvResult: TQValidationResult(
        success: true,
        gpuLayersEnabled: true,
        offloadKv: false,
        cpuKvFallback: true,
        flashAttentionAuto: true,
        flashAttentionRequired: true,
        nGpuLayers: 99,
      ),
      gpuKvResult: TQValidationResult(
        success: false,
        gpuLayersEnabled: true,
        offloadKv: true,
        cpuKvFallback: false,
        flashAttentionAuto: true,
        flashAttentionRequired: true,
        nGpuLayers: 99,
        error: 'Metal pipeline rejected turbo3',
      ),
    );

    expect(option.isAvailable, isTrue);
    expect(option.preferredResult?.pathLabel, 'CPU-KV fallback');
    expect(option.dropdownLabel, 'turbo3 (CPU-KV fallback)');
  });
}
