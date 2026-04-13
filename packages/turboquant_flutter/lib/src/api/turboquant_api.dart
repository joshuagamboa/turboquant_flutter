import 'dart:async';
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
  final bool useGpu;

  TQConfig({
    required this.modelPath,
    this.nCtx = 2048,
    this.nThreads = 4,
    this.cacheTypeK = 'q8_0',
    this.cacheTypeV = 'turbo4',
    this.useGpu = true,
  });
}

class TQProbeResult {
  final bool gpuAvailable;
  final bool metalAvailable;
  final bool vulkanAvailable;
  final bool turbo3Supported;
  final bool turbo4Supported;
  final int systemRamMb;
  final int recommendedNCtx;

  TQProbeResult({
    required this.gpuAvailable,
    required this.metalAvailable,
    required this.vulkanAvailable,
    required this.turbo3Supported,
    required this.turbo4Supported,
    required this.systemRamMb,
    required this.recommendedNCtx,
  });
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
        // Try @rpath for linked framework
        try {
           return DynamicLibrary.open('turboquant_flutter.framework/turboquant_flutter');
        } catch (e) {
           print('DEBUG: Failed to open via turboquant_flutter.framework, trying process(): $e');
           return DynamicLibrary.process();
        }
      }
      if (Platform.isAndroid || Platform.isLinux) {
        return DynamicLibrary.open('libtq_ffi.so');
      }
    } catch (e) {
      print('CRITICAL: Failed to load native library: $e');
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
      gpuAvailable: result['gpuAvailable'],
      metalAvailable: result['metalAvailable'],
      vulkanAvailable: result['vulkanAvailable'],
      turbo3Supported: result['turbo3Supported'],
      turbo4Supported: result['turbo4Supported'],
      systemRamMb: result['systemRamMb'],
      recommendedNCtx: result['recommendedNCtx'],
    );
  }

  static void _probeIsolateEntry(SendPort sendPort) {
    final tq = TurboQuant();
    final bindings = tq._bindings;
    final outProbe = calloc<tq_probe_result_t>();
    final errBuf = calloc<Char>(1024);

    final success = bindings.tq_probe(outProbe, errBuf, 1024);
    if (!success) {
      sendPort.send({'error': errBuf.cast<Utf8>().toDartString()});
    } else {
      sendPort.send({
        'gpuAvailable': outProbe.ref.gpu_available,
        'metalAvailable': outProbe.ref.metal_available,
        'vulkanAvailable': outProbe.ref.vulkan_available,
        'turbo3Supported': outProbe.ref.turbo3_supported,
        'turbo4Supported': outProbe.ref.turbo4_supported,
        'systemRamMb': outProbe.ref.system_ram_mb,
        'recommendedNCtx': outProbe.ref.recommended_n_ctx,
      });
    }

    calloc.free(outProbe);
    calloc.free(errBuf);
  }

  Future<TQGenerationController> generate(TQConfig config, String prompt) async {
    final receivePort = ReceivePort();
    final errorPort = ReceivePort();

    final isolate = await Isolate.spawn(
      _isolateEntry,
      _IsolateParams(
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
      if (!controller.isClosed) controller.close();
    }

    receivePort.listen((message) {
      if (message is TQTokenResponse) {
        if (!controller.isClosed) controller.add(message);
        if (message.isEnd || message.error != null) {
          cleanup();
        }
      }
    });

    errorPort.listen((error) {
      if (!controller.isClosed) {
        controller.add(TQTokenResponse(token: '', isEnd: true, error: error.toString()));
      }
      cleanup();
    });

    return TQGenerationController(
      stream: controller.stream,
      onCancel: cleanup,
    );
  }

  static void _isolateEntry(_IsolateParams params) {
    final tq = TurboQuant();
    final bindings = tq._bindings;

    final errBuf = calloc<Char>(1024);
    final nativeConfig = calloc<tq_config_t>();

    String path = params.config.modelPath;
    print('DEBUG: TurboQuant loading model from path: $path');
    if (path.startsWith('file://')) {
      path = Uri.parse(path).toFilePath();
      print('DEBUG: Stripped file://, new path: $path');
    }
    nativeConfig.ref.model_path = path.toNativeUtf8().cast<Char>();
    nativeConfig.ref.n_ctx = params.config.nCtx;
    nativeConfig.ref.n_threads = params.config.nThreads;
    nativeConfig.ref.cache_type_k = params.config.cacheTypeK.toNativeUtf8().cast<Char>();
    nativeConfig.ref.cache_type_v = params.config.cacheTypeV.toNativeUtf8().cast<Char>();
    nativeConfig.ref.use_gpu = params.config.useGpu;

    final engine = bindings.tq_init(nativeConfig.ref, errBuf, 1024);

    if (engine == nullptr) {
      params.sendPort.send(TQTokenResponse(
        token: '',
        isEnd: true,
        error: errBuf.cast<Utf8>().toDartString(),
      ));
      _cleanupIsolate(nativeConfig, errBuf);
      return;
    }

    final promptPtr = params.prompt.toNativeUtf8().cast<Char>();

    final callback = NativeCallable<Void Function(Pointer<Char>, Bool, Pointer<Void>)>.isolateLocal(
      (Pointer<Char> token, bool isEnd, Pointer<Void> userData) {
        final tokenStr = token.cast<Utf8>().toDartString();
        params.sendPort.send(TQTokenResponse(token: tokenStr, isEnd: isEnd));
      },
    );

    final genOk = bindings.tq_generate(
      engine,
      promptPtr,
      callback.nativeFunction,
      nullptr,
      errBuf,
      1024,
    );

    if (!genOk) {
      final msg = errBuf.cast<Utf8>().toDartString();
      params.sendPort.send(TQTokenResponse(
        token: '',
        isEnd: true,
        error: 'Generate failed: $msg',
      ));
    }

    bindings.tq_free(engine);
    callback.close();
    calloc.free(promptPtr);
    _cleanupIsolate(nativeConfig, errBuf);
  }

  static void _cleanupIsolate(Pointer<tq_config_t> nativeConfig, Pointer<Char> errBuf) {
    calloc.free(errBuf);
    if (nativeConfig.ref.model_path != nullptr) calloc.free(nativeConfig.ref.model_path);
    if (nativeConfig.ref.cache_type_k != nullptr) calloc.free(nativeConfig.ref.cache_type_k);
    if (nativeConfig.ref.cache_type_v != nullptr) calloc.free(nativeConfig.ref.cache_type_v);
    calloc.free(nativeConfig);
  }
}

class _IsolateParams {
  final TQConfig config;
  final String prompt;
  final SendPort sendPort;

  _IsolateParams({
    required this.config,
    required this.prompt,
    required this.sendPort,
  });
}
