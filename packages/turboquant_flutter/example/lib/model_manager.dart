import 'dart:io';
import 'package:dio/dio.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;

class GGUFModel {
  final String name;
  final String url;
  final String filename;

  GGUFModel({required this.name, required this.url, required this.filename});
}

class ModelManager {
  static final List<GGUFModel> testModels = [
    GGUFModel(
      name: 'Gemma 4 E2B (Q4_K_M)',
      url: 'https://huggingface.co/bartowski/google_gemma-4-E2B-it-GGUF/resolve/main/google_gemma-4-E2B-it-Q4_K_M.gguf',
      filename: 'google_gemma-4-E2B-it-Q4_K_M.gguf',
    ),
    GGUFModel(
      name: 'Gemma 4 E2B (Q8_0)',
      url: 'https://huggingface.co/bartowski/google_gemma-4-E2B-it-GGUF/resolve/main/google_gemma-4-E2B-it-Q8_0.gguf',
      filename: 'google_gemma-4-E2B-it-Q8_0.gguf',
    ),
  ];

  Future<String> getLocalPath(String filename) async {
    final directory = await getApplicationDocumentsDirectory();
    return p.join(directory.path, 'models', filename);
  }

  Future<bool> isModelDownloaded(String filename) async {
    final path = await getLocalPath(filename);
    return File(path).existsSync();
  }

  Future<void> downloadModel(GGUFModel model, Function(double) onProgress) async {
    final savePath = await getLocalPath(model.filename);
    final file = File(savePath);
    if (!file.parent.existsSync()) {
      await file.parent.create(recursive: true);
    }

    final dio = Dio();
    await dio.download(
      model.url,
      savePath,
      onReceiveProgress: (received, total) {
        if (total != -1) {
          onProgress(received / total);
        }
      },
    );
  }

  Future<List<File>> getDownloadedModels() async {
    final directory = await getApplicationDocumentsDirectory();
    final modelDir = Directory(p.join(directory.path, 'models'));
    if (!modelDir.existsSync()) return [];
    
    return modelDir
        .listSync()
        .whereType<File>()
        .where((file) => file.path.endsWith('.gguf'))
        .toList();
  }
}
