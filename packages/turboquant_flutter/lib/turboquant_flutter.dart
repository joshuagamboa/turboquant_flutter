
import 'turboquant_flutter_platform_interface.dart';

class TurboquantFlutter {
  Future<String?> getPlatformVersion() {
    return TurboquantFlutterPlatform.instance.getPlatformVersion();
  }
}
