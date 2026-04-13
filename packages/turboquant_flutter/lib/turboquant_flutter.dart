import 'turboquant_flutter_platform_interface.dart';
export 'src/api/turboquant_api.dart';

class TurboquantFlutter {
  Future<String?> getPlatformVersion() {
    return TurboquantFlutterPlatform.instance.getPlatformVersion();
  }
}
