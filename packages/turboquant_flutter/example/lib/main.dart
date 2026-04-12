import 'package:flutter/material.dart';
import 'package:turboquant_flutter/src/api/turboquant_api.dart';
import 'package:file_picker/file_picker.dart';
import 'dart:async';

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
  String? _modelPath;
  final _promptController = TextEditingController(text: 'Tell me a short story about a space pirate.');
  String _response = '';
  bool _isGenerating = false;
  String? _error;

  Future<void> _pickModel() async {
    final result = await FilePicker.platform.pickFiles(
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
        nCtx: 512,
        useGpu: true,
      );

      final stream = await _turboQuant.generate(config, _promptController.text);
      
      await for (final response in stream) {
        if (response.error != null) {
          setState(() {
            _error = response.error;
            _isGenerating = false;
          });
          break;
        }
        
        setState(() {
          _response += response.token;
          if (response.isEnd) {
            _isGenerating = false;
          }
        });
      }
    } catch (e) {
      setState(() {
        _error = e.toString();
        _isGenerating = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          title: const Text('TurboQuant Example'),
        ),
        body: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              ElevatedButton(
                onPressed: _isGenerating ? null : _pickModel,
                child: Text(_modelPath == null ? 'Pick GGUF Model' : 'Model: ${_modelPath!.split('/').last}'),
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
              ElevatedButton(
                onPressed: _isGenerating || _modelPath == null ? null : _generate,
                child: _isGenerating ? const CircularProgressIndicator() : const Text('Generate'),
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
