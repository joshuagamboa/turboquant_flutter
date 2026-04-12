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

class TQTokenResponse {
  final String token;
  final bool isEnd;
  final String? error;

  TQTokenResponse({required this.token, required this.isEnd, this.error});
}

class TurboQuant {
  final DynamicLibrary _lib;
  late final TurboQuantBindings _bindings;

  TurboQuant() : _lib = _loadLibrary() {
    _bindings = TurboQuantBindings(_lib);
  }

  static DynamicLibrary _loadLibrary() {
    if (Platform.environment.containsKey('TQ_FFI_PATH')) {
      return DynamicLibrary.open(Platform.environment['TQ_FFI_PATH']!);
    }
    if (Platform.isMacOS || Platform.isIOS) {
      return DynamicLibrary.process();
    }
    if (Platform.isAndroid || Platform.isLinux) {
      return DynamicLibrary.open('libtq_ffi.so');
    }
    throw UnsupportedError('Unsupported platform: ${Platform.operatingSystem}');
  }

  Future<Stream<TQTokenResponse>> generate(TQConfig config, String prompt) async {
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

    receivePort.listen((message) {
      if (message is TQTokenResponse) {
        controller.add(message);
        if (message.isEnd || message.error != null) {
          receivePort.close();
          errorPort.close();
          isolate.kill(priority: Isolate.immediate);
          controller.close();
        }
      }
    });

    errorPort.listen((error) {
      controller.add(TQTokenResponse(token: '', isEnd: true, error: error.toString()));
      receivePort.close();
      errorPort.close();
      isolate.kill(priority: Isolate.immediate);
      controller.close();
    });

    return controller.stream;
  }

  static void _isolateEntry(_IsolateParams params) {
    final tq = TurboQuant();
    final bindings = tq._bindings;

    final errBuf = calloc<Char>(1024);
    
    final nativeConfig = calloc<tq_config_t>();
    nativeConfig.ref.model_path = params.config.modelPath.toNativeUtf8().cast<Char>();
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
      calloc.free(errBuf);
      calloc.free(nativeConfig.ref.model_path);
      calloc.free(nativeConfig.ref.cache_type_k);
      calloc.free(nativeConfig.ref.cache_type_v);
      calloc.free(nativeConfig);
      return;
    }

    final promptPtr = params.prompt.toNativeUtf8().cast<Char>();

    final callback = NativeCallable<Void Function(Pointer<Char>, Bool, Pointer<Void>)>.isolateLocal(
      (Pointer<Char> token, bool isEnd, Pointer<Void> userData) {
        final tokenStr = token.cast<Utf8>().toDartString();
        params.sendPort.send(TQTokenResponse(token: tokenStr, isEnd: isEnd));
      },
    );

    bindings.tq_generate(
      engine,
      promptPtr,
      callback.nativeFunction,
      nullptr,
      errBuf,
      1024,
    );

    bindings.tq_free(engine);
    
    callback.close();
    calloc.free(promptPtr);
    calloc.free(errBuf);
    calloc.free(nativeConfig.ref.model_path);
    calloc.free(nativeConfig.ref.cache_type_k);
    calloc.free(nativeConfig.ref.cache_type_v);
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
