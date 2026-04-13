import 'package:flutter/material.dart';
import 'package:turboquant_flutter/src/api/turboquant_api.dart';
import 'package:file_picker/file_picker.dart';
import 'dart:async';
import 'dart:io';
import 'package:url_launcher/url_launcher.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> with WidgetsBindingObserver {
  final _turboQuant = TurboQuant();
  
  String? _modelPath;
  final _promptController = TextEditingController(text: 'Tell me a short story about a space pirate.');
  String _response = '';
  bool _isGenerating = false;
  String? _error;
  TQProbeResult? _probeResult;
  TQGenerationController? _generationController;

  String _selectedCacheType = 'turbo4';
  final List<String> _cacheTypes = ['f16', 'q8_0', 'turbo3', 'turbo4'];

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _doProbe();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _stop();
    _promptController.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.paused || state == AppLifecycleState.inactive) {
      if (_isGenerating) {
        _stop();
      }
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
      final file = File(_modelPath!);
      final size = await file.length();
      print('DEBUG: Loading model file of size: $size bytes');

      final config = TQConfig(
        modelPath: _modelPath!,
        nCtx: Platform.isAndroid ? 512 : (_probeResult?.recommendedNCtx ?? 512),
        useGpu: _probeResult?.gpuAvailable ?? true,
        cacheTypeK: 'q8_0',
        cacheTypeV: _selectedCacheType,
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

  Future<void> _launchUrl(String url) async {
    if (!await launchUrl(Uri.parse(url))) {
      setState(() => _error = 'Could not launch $url');
    }
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
              const Text('Download Gemma 4 Models (External):', style: TextStyle(fontWeight: FontWeight.bold)),
              TextButton(
                onPressed: () => _launchUrl('https://huggingface.co/unsloth/gemma-4-2b-it-GGUF'),
                child: const Text('Gemma 4 E2B IT (Unsloth GGUF Repo)'),
              ),
              const Divider(),
              ElevatedButton(
                onPressed: _isGenerating ? null : _pickModel,
                style: ElevatedButton.styleFrom(backgroundColor: Colors.blue, foregroundColor: Colors.white),
                child: Text(_modelPath == null ? 'SELECT LOCAL GGUF MODEL' : 'Model: ${_modelPath!.split('/').last}'),
              ),
              const SizedBox(height: 16),
              DropdownButtonFormField<String>(
                value: _selectedCacheType,
                decoration: const InputDecoration(
                  labelText: 'KV-Cache Type (TurboQuant)',
                  border: OutlineInputBorder(),
                ),
                items: _cacheTypes.map((type) {
                  return DropdownMenuItem(
                    value: type,
                    child: Text(type == 'f16' ? 'f16 (Off)' : type),
                  );
                }).toList(),
                onChanged: _isGenerating ? null : (value) {
                  if (value != null) {
                    setState(() => _selectedCacheType = value);
                  }
                },
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
