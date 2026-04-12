import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:turboquant_flutter_example/main.dart' as app;

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('End-to-End Integration Test: Download, Pick, and Generate', (WidgetTester tester) async {
    app.main();
    await tester.pumpAndSettle();

    // Verify Probe Results are visible
    expect(find.textContaining('GPU:'), findsOneWidget);
    expect(find.textContaining('System RAM:'), findsOneWidget);

    // Find the Q4_K_M model tile
    final q4ModelFinder = find.widgetWithText(ListTile, 'Gemma 4 E2B (Q4_K_M)');
    expect(q4ModelFinder, findsOneWidget);

    // Wait for the download to finish if it's currently downloading
    // or trigger it if not downloaded
    final downloadIconFinder = find.descendant(
      of: q4ModelFinder,
      matching: find.byIcon(Icons.download),
    );
    
    if (downloadIconFinder.evaluate().isNotEmpty) {
      await tester.tap(downloadIconFinder);
      await tester.pump();
      
      // Wait for download to complete (max 5 minutes)
      int waitLoops = 0;
      while (find.descendant(of: q4ModelFinder, matching: find.byIcon(Icons.check_circle)).evaluate().isEmpty && waitLoops < 60) {
        await tester.pump(const Duration(seconds: 5));
        waitLoops++;
      }
      expect(find.descendant(of: q4ModelFinder, matching: find.byIcon(Icons.check_circle)), findsOneWidget, reason: 'Model download timed out');
    }

    // Tap the model to select it
    await tester.tap(q4ModelFinder);
    await tester.pumpAndSettle();

    // Enter a simple prompt
    final promptFinder = find.byType(TextField);
    await tester.enterText(promptFinder, 'Say hi');
    await tester.pumpAndSettle();

    // Tap generate
    final generateButtonFinder = find.widgetWithText(ElevatedButton, 'Generate');
    await tester.tap(generateButtonFinder);
    await tester.pump(); // Start generation

    // Wait for generation to finish or timeout
    int genLoops = 0;
    while (find.textContaining('[Benchmark:').evaluate().isEmpty && genLoops < 30) {
      await tester.pump(const Duration(seconds: 1));
      genLoops++;
    }

    // Check if the benchmark text appeared indicating success
    expect(find.textContaining('[Benchmark:'), findsOneWidget, reason: 'Generation failed or timed out');
  });
}
