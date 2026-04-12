import 'package:flutter/material.dart';
import 'package:turboquant_flutter/src/api/turboquant_api.dart';
import 'package:file_picker/file_picker.dart';
import 'dart:async';
import 'dart:io';
import 'model_manager.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  final _turboQuant = TurboQuant();
  final _modelManager = ModelManager();
  
  String? _modelPath;
  final _promptController = TextEditingController(text: 'Tell me a short story about a space pirate.');
  String _response = '';
  bool _isGenerating = false;
  String? _error;
  TQProbeResult? _probeResult;
  TQGenerationController? _generationController;

  Map<String, double> _downloadProgress = {};
  List<File> _downloadedModels = [];

  @override
  void initState() {
    super.initState();
    _doProbe();
    _refreshModels();
  }

  Future<void> _refreshModels() async {
    final models = await _modelManager.getDownloadedModels();
    setState(() {
      _downloadedModels = models;
    });
  }

  Future<void> _downloadModel(GGUFModel model) async {
    try {
      setState(() => _error = null);
      await _modelManager.downloadModel(model, (progress) {
        setState(() {
          _downloadProgress[model.filename] = progress;
        });
      });
      await _refreshModels();
    } catch (e) {
      setState(() => _error = 'Download failed: $e');
    }
  }

  Future<void> _doProbe() async {
    try {
      final res = await _turboQuant.probe();
      setState(() {
        _probeResult = res;
      });
    } catch (e) {
      setState(() {
        _error = 'Probe failed: $e';
      });
    }
  }

  Future<void> _pickModel() async {
    final result = await FilePicker.pickFiles(
      type: FileType.any,
    );

    if (result != null && result.files.single.path != null) {
      setState(() {
        _modelPath = result.files.single.path;
        _error = null;
      });
    }
  }

  Future<void> _generate() async {
    if (_modelPath == null) {
      setState(() {
        _error = 'Please pick a model first';
      });
      return;
    }

    setState(() {
      _response = '';
      _isGenerating = true;
      _error = null;
    });

    try {
      final config = TQConfig(
        modelPath: _modelPath!,
        nCtx: _probeResult?.recommendedNCtx ?? 512,
        useGpu: _probeResult?.gpuAvailable ?? true,
        cacheTypeK: 'q8_0',
        cacheTypeV: 'turbo4',
      );

      _generationController = await _turboQuant.generate(config, _promptController.text);
      
      final stopwatch = Stopwatch()..start();
      int tokenCount = 0;

      await for (final response in _generationController!.stream) {
        if (response.error != null) {
          setState(() {
            _error = response.error;
            _isGenerating = false;
          });
          break;
        }
        
        tokenCount++;
        setState(() {
          _response += response.token;
          if (response.isEnd) {
            _isGenerating = false;
            _generationController = null;
            stopwatch.stop();
            final tps = tokenCount / (stopwatch.elapsedMilliseconds / 1000);
            _response += '\n\n[Benchmark: ${tps.toStringAsFixed(2)} tokens/sec]';
          }
        });
      }
    } catch (e) {
      setState(() {
        _error = e.toString();
        _isGenerating = false;
        _generationController = null;
      });
    }
  }

  void _stop() {
    _generationController?.cancel();
    setState(() {
      _isGenerating = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          title: const Text('TurboQuant Hardening'),
        ),
        body: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              if (_probeResult != null) ...[
                Text('GPU: ${_probeResult!.gpuAvailable ? "Yes" : "No"} '
                    '(${_probeResult!.metalAvailable ? "Metal" : _probeResult!.vulkanAvailable ? "Vulkan" : "None"})'),
                Text('System RAM: ${_probeResult!.systemRamMb} MB'),
                Text('Recommended n_ctx: ${_probeResult!.recommendedNCtx}'),
                const SizedBox(height: 8),
              ],
              const Divider(),
              const Text('Test Models (Gemma 4 E2B)', style: TextStyle(fontWeight: FontWeight.bold)),
              ...ModelManager.testModels.map((m) {
                final isDownloaded = _downloadedModels.any((f) => f.path.endsWith(m.filename));
                final progress = _downloadProgress[m.filename];
                return ListTile(
                  title: Text(m.name),
                  trailing: isDownloaded 
                    ? const Icon(Icons.check_circle, color: Colors.green)
                    : (progress != null && progress < 1.0)
                      ? SizedBox(width: 24, height: 24, child: CircularProgressIndicator(value: progress))
                      : IconButton(icon: const Icon(Icons.download), onPressed: () => _downloadModel(m)),
                  onTap: isDownloaded ? () async {
                    final path = await _modelManager.getLocalPath(m.filename);
                    setState(() => _modelPath = path);
                  } : null,
                  selected: _modelPath?.endsWith(m.filename) ?? false,
                );
              }),
              const Divider(),
              ElevatedButton(
                onPressed: _isGenerating ? null : _pickModel,
                child: Text(_modelPath == null ? 'Or Pick Local GGUF' : 'Model: ${_modelPath!.split('/').last}'),
              ),
              const SizedBox(height: 16),
              TextField(
                controller: _promptController,
                decoration: const InputDecoration(
                  labelText: 'Prompt',
                  border: OutlineInputBorder(),
                ),
                maxLines: 3,
              ),
              const SizedBox(height: 16),
              Row(
                children: [
                  Expanded(
                    child: ElevatedButton(
                      onPressed: _isGenerating || _modelPath == null ? null : _generate,
                      child: _isGenerating ? const SizedBox(height: 20, width: 20, child: CircularProgressIndicator(strokeWidth: 2)) : const Text('Generate'),
                    ),
                  ),
                  if (_isGenerating) ...[
                    const SizedBox(width: 8),
                    ElevatedButton(
                      onPressed: _stop,
                      style: ElevatedButton.styleFrom(backgroundColor: Colors.red, foregroundColor: Colors.white),
                      child: const Text('Stop'),
                    ),
                  ],
                ],
              ),
              if (_error != null) ...[
                const SizedBox(height: 16),
                Text(
                  _error!,
                  style: const TextStyle(color: Colors.red),
                ),
              ],
              const SizedBox(height: 16),
              const Text(
                'Response:',
                style: TextStyle(fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 8),
              Expanded(
                child: Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    border: Border.all(color: Colors.grey),
                    borderRadius: BorderRadius.circular(4),
                  ),
                  child: SingleChildScrollView(
                    child: Text(_response),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
