import 'dart:io';
import 'package:turboquant_flutter/src/api/turboquant_api.dart';
import 'package:test/test.dart';

void main() {
  test('TurboQuant FFI Load and Generate (Dry Run - expects failure if no model)', () async {
    final tq = TurboQuant();
    final config = TQConfig(
      modelPath: 'non_existent_model.gguf',
      nCtx: 512,
      useGpu: false,
    );

    final stream = await tq.generate(config, 'Hello');
    
    await for (final response in stream) {
      if (response.error != null) {
        print('Expected error (no model): ${response.error}');
        expect(response.error, contains('Failed to load model'));
      } else {
        print('Token: ${response.token}');
      }
    }
  });
}
