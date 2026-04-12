import 'dart:io';
import 'package:turboquant_flutter/src/api/turboquant_api.dart';
import 'package:test/test.dart';

void main() {
  test('TurboQuant Probe', () async {
    final tq = TurboQuant();
    final res = await tq.probe();
    print('GPU Available: ${res.gpuAvailable}');
    print('Metal Available: ${res.metalAvailable}');
    print('Vulkan Available: ${res.vulkanAvailable}');
    print('Turbo4 Supported: ${res.turbo4Supported}');
    print('Recommended n_ctx: ${res.recommendedNCtx}');
    
    expect(res.recommendedNCtx, greaterThan(0));
  });

  test('TurboQuant FFI Load and Generate (Dry Run - expects failure if no model)', () async {
    final tq = TurboQuant();
    final config = TQConfig(
      modelPath: 'non_existent_model.gguf',
      nCtx: 512,
      useGpu: false,
    );

    final controller = await tq.generate(config, 'Hello');
    
    await for (final response in controller.stream) {
      if (response.error != null) {
        print('Expected error (no model): ${response.error}');
        expect(response.error, contains('Failed to load model'));
      } else {
        print('Token: ${response.token}');
      }
    }
  });
}
