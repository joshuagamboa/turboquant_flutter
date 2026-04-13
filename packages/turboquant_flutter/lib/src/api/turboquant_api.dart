import 'dart:async';
import 'dart:developer' as developer;
import 'dart:ffi';
import 'dart:io';
import 'dart:isolate';

import 'package:ffi/ffi.dart';

import '../ffi/turboquant_bindings.dart';

class TQConfig {
  final String modelPath;
  final int nCtx;
  final int nThreads;
  final String cacheTypeK;
  final String cacheTypeV;
  final int nGpuLayers;
  final bool offloadKv;

  TQConfig({
    required this.modelPath,
    this.nCtx = 2048,
    this.nThreads = 4,
    this.cacheTypeK = 'q8_0',
    this.cacheTypeV = 'turbo4',
    int? nGpuLayers,
    bool? useGpu,
    this.offloadKv = false,
  }) : nGpuLayers = nGpuLayers ?? ((useGpu ?? true) ? 99 : 0);

  bool get useGpuLayers => nGpuLayers > 0;
}

class TQProbeResult {
  final bool gpuAvailable;
  final bool metalAvailable;
  final bool vulkanAvailable;
  final bool isSimulator;
  final bool turbo3Supported;
  final bool turbo4Supported;
  final bool simdgroupReductionAvailable;
  final bool tensorApiAvailable;
  final int appleGpuFamily;
  final int systemRamMb;
  final int recommendedNCtx;

  TQProbeResult({
    required this.gpuAvailable,
    required this.metalAvailable,
    required this.vulkanAvailable,
    required this.isSimulator,
    required this.turbo3Supported,
    required this.turbo4Supported,
    required this.simdgroupReductionAvailable,
    required this.tensorApiAvailable,
    required this.appleGpuFamily,
    required this.systemRamMb,
    required this.recommendedNCtx,
  });
}

class TQValidationResult {
  final bool success;
  final bool gpuLayersEnabled;
  final bool offloadKv;
  final bool cpuKvFallback;
  final bool flashAttentionAuto;
  final bool flashAttentionRequired;
  final int nGpuLayers;
  final String? error;

  TQValidationResult({
    required this.success,
    required this.gpuLayersEnabled,
    required this.offloadKv,
    required this.cpuKvFallback,
    required this.flashAttentionAuto,
    required this.flashAttentionRequired,
    required this.nGpuLayers,
    this.error,
  });

  String get pathLabel {
    if (gpuLayersEnabled && offloadKv) {
      return 'GPU-KV';
    }
    if (cpuKvFallback) {
      return 'CPU-KV fallback';
    }
    return 'CPU-only';
  }
}

class TQTokenResponse {
  final String token;
  final bool isEnd;
  final String? error;

  TQTokenResponse({required this.token, required this.isEnd, this.error});
}

class TQGenerationController {
  final Stream<TQTokenResponse> stream;
  final void Function() onCancel;

  TQGenerationController({required this.stream, required this.onCancel});

  void cancel() => onCancel();
}

class TurboQuant {
  final DynamicLibrary _lib;
  late final TurboQuantBindings _bindings;

  TurboQuant() : _lib = _loadLibrary() {
    _bindings = TurboQuantBindings(_lib);
  }

  static DynamicLibrary _loadLibrary() {
    try {
      if (Platform.environment.containsKey('TQ_FFI_PATH')) {
        return DynamicLibrary.open(Platform.environment['TQ_FFI_PATH']!);
      }
      if (Platform.isMacOS) {
        return DynamicLibrary.process();
      }
      if (Platform.isIOS) {
        try {
          return DynamicLibrary.open(
            'turboquant_flutter.framework/turboquant_flutter',
          );
        } catch (e) {
          developer.log(
            'Failed to open turboquant_flutter.framework, falling back to process()',
            name: 'turboquant_flutter',
            error: e,
          );
          return DynamicLibrary.process();
        }
      }
      if (Platform.isAndroid || Platform.isLinux) {
        return DynamicLibrary.open('libtq_ffi.so');
      }
    } catch (e) {
      developer.log(
        'Failed to load native library',
        name: 'turboquant_flutter',
        error: e,
      );
      rethrow;
    }
    throw UnsupportedError('Unsupported platform: ${Platform.operatingSystem}');
  }

  Future<TQProbeResult> probe() async {
    final receivePort = ReceivePort();
    await Isolate.spawn(_probeIsolateEntry, receivePort.sendPort);
    final result = await receivePort.first as Map<String, dynamic>;

    if (result['error'] != null) {
      throw Exception(result['error']);
    }

    return TQProbeResult(
      gpuAvailable: result['gpuAvailable'] as bool,
      metalAvailable: result['metalAvailable'] as bool,
      vulkanAvailable: result['vulkanAvailable'] as bool,
      isSimulator: result['isSimulator'] as bool,
      turbo3Supported: result['turbo3Supported'] as bool,
      turbo4Supported: result['turbo4Supported'] as bool,
      simdgroupReductionAvailable:
          result['simdgroupReductionAvailable'] as bool,
      tensorApiAvailable: result['tensorApiAvailable'] as bool,
      appleGpuFamily: result['appleGpuFamily'] as int,
      systemRamMb: result['systemRamMb'] as int,
      recommendedNCtx: result['recommendedNCtx'] as int,
    );
  }

  Future<TQValidationResult> validateConfig(TQConfig config) async {
    final receivePort = ReceivePort();
    await Isolate.spawn(
      _validateIsolateEntry,
      _ValidationParams(config: config, sendPort: receivePort.sendPort),
    );
    final result = await receivePort.first as Map<String, dynamic>;

    return TQValidationResult(
      success: result['success'] as bool,
      gpuLayersEnabled: result['gpuLayersEnabled'] as bool,
      offloadKv: result['offloadKv'] as bool,
      cpuKvFallback: result['cpuKvFallback'] as bool,
      flashAttentionAuto: result['flashAttentionAuto'] as bool,
      flashAttentionRequired: result['flashAttentionRequired'] as bool,
      nGpuLayers: result['nGpuLayers'] as int,
      error: result['error'] as String?,
    );
  }

  static void _probeIsolateEntry(SendPort sendPort) {
    final tq = TurboQuant();
    final bindings = tq._bindings;
    final outProbe = calloc<tq_probe_result_t>();
    final errBuf = calloc<Char>(2048);

    final success = bindings.tq_probe(outProbe, errBuf, 2048);
    if (!success) {
      sendPort.send({'error': errBuf.cast<Utf8>().toDartString()});
    } else {
      sendPort.send({
        'gpuAvailable': outProbe.ref.gpu_available,
        'metalAvailable': outProbe.ref.metal_available,
        'vulkanAvailable': outProbe.ref.vulkan_available,
        'isSimulator': outProbe.ref.is_simulator,
        'turbo3Supported': outProbe.ref.turbo3_supported,
        'turbo4Supported': outProbe.ref.turbo4_supported,
        'simdgroupReductionAvailable':
            outProbe.ref.simdgroup_reduction_available,
        'tensorApiAvailable': outProbe.ref.tensor_api_available,
        'appleGpuFamily': outProbe.ref.apple_gpu_family,
        'systemRamMb': outProbe.ref.system_ram_mb,
        'recommendedNCtx': outProbe.ref.recommended_n_ctx,
      });
    }

    calloc.free(outProbe);
    calloc.free(errBuf);
  }

  static void _validateIsolateEntry(_ValidationParams params) {
    final tq = TurboQuant();
    final bindings = tq._bindings;
    final errBuf = calloc<Char>(4096);
    final nativeConfig = _NativeConfigHandle.fromConfig(params.config);
    final outResult = calloc<tq_validation_result_t>();

    final success = bindings.tq_validate_config(
      nativeConfig.pointer.ref,
      outResult,
      errBuf,
      4096,
    );

    params.sendPort.send({
      'success': success && outResult.ref.success,
      'gpuLayersEnabled': outResult.ref.gpu_layers_enabled,
      'offloadKv': outResult.ref.offload_kv,
      'cpuKvFallback': outResult.ref.cpu_kv_fallback,
      'flashAttentionAuto': outResult.ref.flash_attention_auto,
      'flashAttentionRequired': outResult.ref.flash_attention_required,
      'nGpuLayers': outResult.ref.n_gpu_layers,
      'error': success ? null : errBuf.cast<Utf8>().toDartString(),
    });

    calloc.free(outResult);
    nativeConfig.dispose();
    calloc.free(errBuf);
  }

  Future<TQGenerationController> generate(
    TQConfig config,
    String prompt,
  ) async {
    final receivePort = ReceivePort();
    final errorPort = ReceivePort();

    final isolate = await Isolate.spawn(
      _generationIsolateEntry,
      _GenerationParams(
        config: config,
        prompt: prompt,
        sendPort: receivePort.sendPort,
      ),
      onError: errorPort.sendPort,
    );

    final controller = StreamController<TQTokenResponse>();

    void cleanup() {
      receivePort.close();
      errorPort.close();
      isolate.kill(priority: Isolate.immediate);
      if (!controller.isClosed) {
        controller.close();
      }
    }

    receivePort.listen((message) {
      if (message is TQTokenResponse) {
        if (!controller.isClosed) {
          controller.add(message);
        }
        if (message.isEnd || message.error != null) {
          cleanup();
        }
      }
    });

    errorPort.listen((error) {
      if (!controller.isClosed) {
        controller.add(
          TQTokenResponse(token: '', isEnd: true, error: error.toString()),
        );
      }
      cleanup();
    });

    return TQGenerationController(stream: controller.stream, onCancel: cleanup);
  }

  static void _generationIsolateEntry(_GenerationParams params) {
    final tq = TurboQuant();
    final bindings = tq._bindings;
    final errBuf = calloc<Char>(4096);
    final nativeConfig = _NativeConfigHandle.fromConfig(params.config);

    final engine = bindings.tq_init(nativeConfig.pointer.ref, errBuf, 4096);

    if (engine == nullptr) {
      params.sendPort.send(
        TQTokenResponse(
          token: '',
          isEnd: true,
          error: errBuf.cast<Utf8>().toDartString(),
        ),
      );
      nativeConfig.dispose();
      calloc.free(errBuf);
      return;
    }

    final promptPtr = params.prompt.toNativeUtf8().cast<Char>();
    final callback =
        NativeCallable<
          Void Function(Pointer<Char>, Bool, Pointer<Void>)
        >.isolateLocal((
          Pointer<Char> token,
          bool isEnd,
          Pointer<Void> userData,
        ) {
          final tokenStr = token.cast<Utf8>().toDartString();
          params.sendPort.send(TQTokenResponse(token: tokenStr, isEnd: isEnd));
        });

    final genOk = bindings.tq_generate(
      engine,
      promptPtr,
      callback.nativeFunction,
      nullptr,
      errBuf,
      4096,
    );

    if (!genOk) {
      params.sendPort.send(
        TQTokenResponse(
          token: '',
          isEnd: true,
          error: 'Generate failed: ${errBuf.cast<Utf8>().toDartString()}',
        ),
      );
    }

    bindings.tq_free(engine);
    callback.close();
    calloc.free(promptPtr);
    nativeConfig.dispose();
    calloc.free(errBuf);
  }
}

class _GenerationParams {
  final TQConfig config;
  final String prompt;
  final SendPort sendPort;

  _GenerationParams({
    required this.config,
    required this.prompt,
    required this.sendPort,
  });
}

class _ValidationParams {
  final TQConfig config;
  final SendPort sendPort;

  _ValidationParams({required this.config, required this.sendPort});
}

class _NativeConfigHandle {
  final Pointer<tq_config_t> pointer;

  _NativeConfigHandle._(this.pointer);

  factory _NativeConfigHandle.fromConfig(TQConfig config) {
    final pointer = calloc<tq_config_t>();
    final normalizedPath = _normalizeModelPath(config.modelPath);
    pointer.ref.model_path = normalizedPath.toNativeUtf8().cast<Char>();
    pointer.ref.n_ctx = config.nCtx;
    pointer.ref.n_threads = config.nThreads;
    pointer.ref.cache_type_k = config.cacheTypeK.toNativeUtf8().cast<Char>();
    pointer.ref.cache_type_v = config.cacheTypeV.toNativeUtf8().cast<Char>();
    pointer.ref.n_gpu_layers = config.nGpuLayers;
    pointer.ref.offload_kv = config.offloadKv;
    return _NativeConfigHandle._(pointer);
  }

  void dispose() {
    if (pointer.ref.model_path != nullptr) {
      calloc.free(pointer.ref.model_path);
    }
    if (pointer.ref.cache_type_k != nullptr) {
      calloc.free(pointer.ref.cache_type_k);
    }
    if (pointer.ref.cache_type_v != nullptr) {
      calloc.free(pointer.ref.cache_type_v);
    }
    calloc.free(pointer);
  }

  static String _normalizeModelPath(String path) {
    if (path.startsWith('file://')) {
      return Uri.parse(path).toFilePath();
    }
    return path;
  }
}
