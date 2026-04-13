import 'package:turboquant_flutter/src/api/turboquant_api.dart';
import 'package:test/test.dart';

void main() {
  test('TQConfig splits GPU layers from KV offload', () {
    final cpuKvFallback = TQConfig(
      modelPath: 'model.gguf',
      nGpuLayers: 99,
      offloadKv: false,
    );
    final cpuOnly = TQConfig(
      modelPath: 'model.gguf',
      useGpu: false,
      offloadKv: false,
    );

    expect(cpuKvFallback.useGpuLayers, isTrue);
    expect(cpuKvFallback.offloadKv, isFalse);
    expect(cpuOnly.useGpuLayers, isFalse);
    expect(cpuOnly.nGpuLayers, 0);
  });

  test('TurboQuant Probe', () async {
    final tq = TurboQuant();
    final res = await tq.probe();

    expect(res.recommendedNCtx, greaterThan(0));
  });

  test(
    'TurboQuant FFI Load and Generate (Dry Run - expects failure if no model)',
    () async {
      final tq = TurboQuant();
      final config = TQConfig(
        modelPath: 'non_existent_model.gguf',
        nCtx: 512,
        useGpu: false,
      );

      final controller = await tq.generate(config, 'Hello');

      await for (final response in controller.stream) {
        if (response.error != null) {
          expect(response.error, contains('File not found or not readable'));
        }
      }
    },
  );
}
