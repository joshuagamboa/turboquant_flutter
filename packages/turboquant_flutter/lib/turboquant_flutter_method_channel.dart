import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import 'turboquant_flutter_platform_interface.dart';

/// An implementation of [TurboquantFlutterPlatform] that uses method channels.
class MethodChannelTurboquantFlutter extends TurboquantFlutterPlatform {
  /// The method channel used to interact with the native platform.
  @visibleForTesting
  final methodChannel = const MethodChannel('turboquant_flutter');

  @override
  Future<String?> getPlatformVersion() async {
    final version = await methodChannel.invokeMethod<String>(
      'getPlatformVersion',
    );
    return version;
  }
}
