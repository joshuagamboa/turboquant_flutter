import 'package:flutter_test/flutter_test.dart';

import 'package:turboquant_flutter_example/main.dart';

void main() {
  testWidgets('renders probe-first validation workflow', (
    WidgetTester tester,
  ) async {
    await tester.pumpWidget(const MyApp());

    expect(find.text('TurboQuant iOS Validation'), findsOneWidget);
    expect(find.text('Run Hardware Probe'), findsOneWidget);
    expect(find.text('Response:'), findsOneWidget);
  });
}
