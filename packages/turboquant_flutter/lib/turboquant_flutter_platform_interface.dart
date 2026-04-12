import 'package:plugin_platform_interface/plugin_platform_interface.dart';

import 'turboquant_flutter_method_channel.dart';

abstract class TurboquantFlutterPlatform extends PlatformInterface {
  /// Constructs a TurboquantFlutterPlatform.
  TurboquantFlutterPlatform() : super(token: _token);

  static final Object _token = Object();

  static TurboquantFlutterPlatform _instance = MethodChannelTurboquantFlutter();

  /// The default instance of [TurboquantFlutterPlatform] to use.
  ///
  /// Defaults to [MethodChannelTurboquantFlutter].
  static TurboquantFlutterPlatform get instance => _instance;

  /// Platform-specific implementations should set this with their own
  /// platform-specific class that extends [TurboquantFlutterPlatform] when
  /// they register themselves.
  static set instance(TurboquantFlutterPlatform instance) {
    PlatformInterface.verifyToken(instance, _token);
    _instance = instance;
  }

  Future<String?> getPlatformVersion() {
    throw UnimplementedError('platformVersion() has not been implemented.');
  }
}
