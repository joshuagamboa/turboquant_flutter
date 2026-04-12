import 'package:flutter_test/flutter_test.dart';
import 'package:turboquant_flutter/turboquant_flutter.dart';
import 'package:turboquant_flutter/turboquant_flutter_platform_interface.dart';
import 'package:turboquant_flutter/turboquant_flutter_method_channel.dart';
import 'package:plugin_platform_interface/plugin_platform_interface.dart';

class MockTurboquantFlutterPlatform
    with MockPlatformInterfaceMixin
    implements TurboquantFlutterPlatform {
  @override
  Future<String?> getPlatformVersion() => Future.value('42');
}

void main() {
  final TurboquantFlutterPlatform initialPlatform = TurboquantFlutterPlatform.instance;

  test('$MethodChannelTurboquantFlutter is the default instance', () {
    expect(initialPlatform, isInstanceOf<MethodChannelTurboquantFlutter>());
  });

  test('getPlatformVersion', () async {
    TurboquantFlutter turboquantFlutterPlugin = TurboquantFlutter();
    MockTurboquantFlutterPlatform fakePlatform = MockTurboquantFlutterPlatform();
    TurboquantFlutterPlatform.instance = fakePlatform;

    expect(await turboquantFlutterPlugin.getPlatformVersion(), '42');
  });
}
