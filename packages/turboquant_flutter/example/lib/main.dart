import 'dart:async';
import 'dart:io';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:turboquant_flutter/turboquant_flutter.dart';
import 'package:url_launcher/url_launcher.dart';

void main() {
  runApp(const MyApp());
}

class KvOptionAvailability {
  final String cacheType;
  final TQValidationResult? cpuKvResult;
  final TQValidationResult? gpuKvResult;

  const KvOptionAvailability({
    required this.cacheType,
    this.cpuKvResult,
    this.gpuKvResult,
  });

  bool get isCompatibilityOption => cacheType == 'f16' || cacheType == 'q8_0';

  TQValidationResult? get preferredResult {
    if (gpuKvResult?.success ?? false) {
      return gpuKvResult;
    }
    if (cpuKvResult?.success ?? false) {
      return cpuKvResult;
    }
    return null;
  }

  bool get isAvailable => isCompatibilityOption || preferredResult != null;

  String get dropdownLabel {
    if (cacheType == 'f16') {
      return 'f16 V (compat)';
    }
    if (cacheType == 'q8_0') {
      return 'q8_0 V (compat)';
    }
    final preferred = preferredResult;
    if (preferred == null) {
      return '$cacheType (validation failed)';
    }
    if (preferred.offloadKv) {
      return '$cacheType (GPU-KV validated)';
    }
    return '$cacheType (${preferred.pathLabel})';
  }
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> with WidgetsBindingObserver {
  static const List<String> _compatibilityCacheTypes = ['f16', 'q8_0'];
  static const List<String> _turboCacheTypes = ['turbo4', 'turbo3'];

  late final TurboQuant _turboQuant;

  bool _turboQuantInitialized = false;
  bool _isGenerating = false;
  bool _isValidating = false;
  bool _cacheTypeChosenByUser = false;

  String? _modelPath;
  String _selectedCacheType = 'q8_0';
  String _response = '';
  String? _error;
  String? _validationError;

  final TextEditingController _promptController = TextEditingController(
    text: 'Tell me a short story about a space pirate.',
  );

  TQProbeResult? _probeResult;
  TQGenerationController? _generationController;
  List<File> _localModels = [];
  Map<String, KvOptionAvailability> _kvOptions = _baseKvOptions();

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _scanModels();
    });
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _stop(updateState: false);
    _promptController.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.paused ||
        state == AppLifecycleState.inactive) {
      if (_isGenerating) {
        _stop();
      }
    }
  }

  static Map<String, KvOptionAvailability> _baseKvOptions() {
    return <String, KvOptionAvailability>{
      'f16': const KvOptionAvailability(cacheType: 'f16'),
      'q8_0': const KvOptionAvailability(cacheType: 'q8_0'),
    };
  }

  Future<void> _initTurboQuant() async {
    if (_turboQuantInitialized) {
      return;
    }

    try {
      _turboQuant = TurboQuant();
      _turboQuantInitialized = true;
    } catch (e) {
      if (!mounted) {
        return;
      }
      setState(() => _error = 'Init failed: $e');
    }
  }

  Future<void> _scanModels() async {
    try {
      final directory = await getApplicationDocumentsDirectory();
      final List<File> allFiles = [];

      if (await directory.exists()) {
        allFiles.addAll(
          directory.listSync().whereType<File>().where(
            (file) => file.path.endsWith('.gguf'),
          ),
        );
      }

      final modelDir = Directory('${directory.path}/models');
      if (await modelDir.exists()) {
        allFiles.addAll(
          modelDir.listSync().whereType<File>().where(
            (file) => file.path.endsWith('.gguf'),
          ),
        );
      }

      if (!mounted) {
        return;
      }

      setState(() {
        _localModels = allFiles;
        if (_modelPath == null && allFiles.isNotEmpty) {
          _modelPath = allFiles.first.path;
        }
      });

      if (_modelPath != null && _probeResult != null) {
        await _refreshValidationResults();
      }
    } catch (e) {
      if (!mounted) {
        return;
      }
      setState(() => _error = 'Scan failed: $e');
    }
  }

  Future<void> _doProbe() async {
    await _initTurboQuant();
    if (!_turboQuantInitialized) {
      return;
    }

    try {
      final result = await _turboQuant.probe();
      if (!mounted) {
        return;
      }
      setState(() {
        _probeResult = result;
        _error = null;
      });
      await _refreshValidationResults();
    } catch (e) {
      if (!mounted) {
        return;
      }
      setState(() {
        _error = 'Probe failed: $e';
      });
    }
  }

  Future<void> _pickModel() async {
    final result = await FilePicker.pickFiles(type: FileType.any);
    final path = result?.files.single.path;
    if (path == null) {
      return;
    }
    await _selectModel(path);
  }

  Future<void> _selectModel(String path) async {
    if (!mounted) {
      return;
    }
    setState(() {
      _modelPath = path;
      _error = null;
      _validationError = null;
      _cacheTypeChosenByUser = false;
    });

    if (_probeResult != null) {
      await _refreshValidationResults();
    }
  }

  bool get _canAttemptGpuKv {
    final probe = _probeResult;
    return probe != null && probe.gpuAvailable && !probe.isSimulator;
  }

  TQConfig _buildConfigForCacheType(
    String cacheType, {
    required bool offloadKv,
  }) {
    final probe = _probeResult;
    final bool gpuLayersEnabled =
        probe != null && probe.gpuAvailable && !probe.isSimulator;
    final int nCtx = probe == null
        ? 1024
        : (probe.isSimulator ? 128 : probe.recommendedNCtx);

    return TQConfig(
      modelPath: _modelPath!,
      nCtx: nCtx,
      nGpuLayers: gpuLayersEnabled ? 99 : 0,
      offloadKv: gpuLayersEnabled ? offloadKv : false,
      cacheTypeK: 'q8_0',
      cacheTypeV: cacheType,
    );
  }

  Future<void> _refreshValidationResults() async {
    if (_modelPath == null || _probeResult == null) {
      if (!mounted) {
        return;
      }
      setState(() {
        _kvOptions = _baseKvOptions();
        _validationError = null;
      });
      return;
    }

    await _initTurboQuant();
    if (!_turboQuantInitialized) {
      return;
    }

    if (!mounted) {
      return;
    }
    setState(() {
      _isValidating = true;
      _validationError = null;
    });

    try {
      final nextOptions = <String, KvOptionAvailability>{..._baseKvOptions()};

      for (final cacheType in _turboCacheTypes) {
        final cpuKvResult = await _turboQuant.validateConfig(
          _buildConfigForCacheType(cacheType, offloadKv: false),
        );

        TQValidationResult? gpuKvResult;
        if (_canAttemptGpuKv) {
          gpuKvResult = await _turboQuant.validateConfig(
            _buildConfigForCacheType(cacheType, offloadKv: true),
          );
        }

        nextOptions[cacheType] = KvOptionAvailability(
          cacheType: cacheType,
          cpuKvResult: cpuKvResult,
          gpuKvResult: gpuKvResult,
        );
      }

      var nextSelection = _selectedCacheType;
      if (!(nextOptions[nextSelection]?.isAvailable ?? false)) {
        nextSelection = 'q8_0';
      }
      if (!_cacheTypeChosenByUser &&
          (nextOptions['turbo4']?.preferredResult != null)) {
        nextSelection = 'turbo4';
      } else if (!_cacheTypeChosenByUser &&
          (nextOptions['turbo3']?.preferredResult != null)) {
        nextSelection = 'turbo3';
      }

      if (!mounted) {
        return;
      }
      setState(() {
        _kvOptions = nextOptions;
        _selectedCacheType = nextSelection;
        _isValidating = false;
      });
    } catch (e) {
      if (!mounted) {
        return;
      }
      setState(() {
        _kvOptions = _baseKvOptions();
        _validationError = 'Validation failed: $e';
        _selectedCacheType = 'q8_0';
        _isValidating = false;
      });
    }
  }

  TQConfig _buildGenerationConfig() {
    final availability = _kvOptions[_selectedCacheType];
    final bool useValidatedGpuKv = availability?.gpuKvResult?.success ?? false;
    final bool offloadKv = _turboCacheTypes.contains(_selectedCacheType)
        ? useValidatedGpuKv
        : _canAttemptGpuKv;

    return _buildConfigForCacheType(_selectedCacheType, offloadKv: offloadKv);
  }

  String _effectiveRuntimeDescription() {
    if (_modelPath == null) {
      return 'Select a model to compute the runtime path.';
    }

    final config = _buildGenerationConfig();
    final availability = _kvOptions[_selectedCacheType];
    final String pathLabel;
    if (_turboCacheTypes.contains(_selectedCacheType)) {
      pathLabel = availability?.preferredResult?.pathLabel ?? 'Unvalidated';
    } else if (config.offloadKv) {
      pathLabel = 'GPU-KV';
    } else if (config.useGpuLayers) {
      pathLabel = 'CPU-KV fallback';
    } else {
      pathLabel = 'CPU-only';
    }

    return 'K=q8_0, V=$_selectedCacheType, GPU layers=${config.nGpuLayers}, '
        'KV offload=${config.offloadKv ? "on" : "off"}, path=$pathLabel';
  }

  Future<void> _generate() async {
    if (_modelPath == null) {
      setState(() => _error = 'Please pick a model first');
      return;
    }

    if (_probeResult == null) {
      await _doProbe();
    }

    if (_probeResult == null) {
      return;
    }

    await _initTurboQuant();
    if (!_turboQuantInitialized) {
      return;
    }

    if (!mounted) {
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
      debugPrint('DEBUG: Loading model file of size: $size bytes');

      final config = _buildGenerationConfig();
      _generationController = await _turboQuant.generate(
        config,
        _promptController.text,
      );

      final stopwatch = Stopwatch()..start();
      int tokenCount = 0;

      await for (final response in _generationController!.stream) {
        if (response.error != null) {
          if (!mounted) {
            return;
          }
          setState(() {
            _error = response.error;
            _isGenerating = false;
          });
          break;
        }

        tokenCount++;
        if (!mounted) {
          return;
        }
        setState(() {
          _response += response.token;
          if (response.isEnd) {
            _isGenerating = false;
            _generationController = null;
            stopwatch.stop();
            if (stopwatch.elapsedMilliseconds > 0) {
              final tps = tokenCount / (stopwatch.elapsedMilliseconds / 1000);
              _response +=
                  '\n\n[Benchmark: ${tps.toStringAsFixed(2)} tokens/sec]';
            }
          }
        });
      }
    } catch (e) {
      if (!mounted) {
        return;
      }
      setState(() {
        _error = e.toString();
        _isGenerating = false;
        _generationController = null;
      });
    }
  }

  void _stop({bool updateState = true}) {
    _generationController?.cancel();
    if (updateState && mounted) {
      setState(() {
        _isGenerating = false;
      });
    }
  }

  Future<void> _launchUrl(String url) async {
    if (!await launchUrl(Uri.parse(url))) {
      if (!mounted) {
        return;
      }
      setState(() => _error = 'Could not launch $url');
    }
  }

  List<String> get _availableCacheTypes {
    final available = <String>[
      ..._compatibilityCacheTypes,
      for (final cacheType in _turboCacheTypes)
        if (_kvOptions[cacheType]?.isAvailable ?? false) cacheType,
    ];
    if (!available.contains(_selectedCacheType)) {
      return <String>['q8_0', ...available.where((type) => type != 'q8_0')];
    }
    return available;
  }

  Widget _buildProbeCard() {
    final probe = _probeResult;
    if (probe == null) {
      return Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          ElevatedButton(
            onPressed: _doProbe,
            child: const Text('Run Hardware Probe'),
          ),
          const SizedBox(height: 8),
          const Text(
            'Run the probe to discover Metal/Vulkan availability and validate TurboQuant paths.',
          ),
        ],
      );
    }

    final backendLabel = probe.metalAvailable
        ? 'Metal'
        : (probe.vulkanAvailable ? 'Vulkan' : 'CPU only');
    final appleFamilyLabel = probe.appleGpuFamily > 0
        ? 'Apple${probe.appleGpuFamily}'
        : 'Unknown';

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('GPU: ${probe.gpuAvailable ? "Yes" : "No"} ($backendLabel)'),
        Text('Simulator: ${probe.isSimulator ? "Yes" : "No"}'),
        Text('Apple GPU family: $appleFamilyLabel'),
        Text(
          'simdgroup reduction: ${probe.simdgroupReductionAvailable ? "Yes" : "No"}',
        ),
        Text('tensor API: ${probe.tensorApiAvailable ? "Yes" : "No"}'),
        Text('System RAM: ${probe.systemRamMb} MB'),
        Text('Recommended n_ctx: ${probe.recommendedNCtx}'),
      ],
    );
  }

  Widget _buildValidationSection() {
    final turboOptions = _turboCacheTypes
        .map((cacheType) => _kvOptions[cacheType])
        .whereType<KvOptionAvailability>()
        .toList();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'TurboQuant validation',
          style: TextStyle(fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 8),
        if (_validationError != null)
          Text(_validationError!, style: const TextStyle(color: Colors.red)),
        if (_isValidating)
          const Padding(
            padding: EdgeInsets.only(bottom: 8),
            child: LinearProgressIndicator(),
          ),
        if (_probeResult == null || _modelPath == null)
          const Text(
            'Select a model and run the hardware probe to validate turbo3/turbo4.',
          )
        else
          ...turboOptions.map((option) {
            final cpuLabel = option.cpuKvResult == null
                ? 'Not run'
                : option.cpuKvResult!.success
                ? 'OK (${option.cpuKvResult!.pathLabel})'
                : option.cpuKvResult!.error ?? 'Failed';
            final gpuLabel = option.gpuKvResult == null
                ? (_canAttemptGpuKv ? 'Not run' : 'Skipped (no GPU-KV path)')
                : option.gpuKvResult!.success
                ? 'OK (${option.gpuKvResult!.pathLabel})'
                : option.gpuKvResult!.error ?? 'Failed';

            return Card(
              margin: const EdgeInsets.only(bottom: 8),
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      option.dropdownLabel,
                      style: const TextStyle(fontWeight: FontWeight.w600),
                    ),
                    const SizedBox(height: 4),
                    Text('CPU-KV fallback: $cpuLabel'),
                    Text('GPU-KV: $gpuLabel'),
                  ],
                ),
              ),
            );
          }),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    final availableCacheTypes = _availableCacheTypes;

    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('TurboQuant iOS Validation')),
        body: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Expanded(
                child: SingleChildScrollView(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      _buildProbeCard(),
                      const SizedBox(height: 12),
                      const Divider(),
                      const Text(
                        'Download Gemma 4 Models (External)',
                        style: TextStyle(fontWeight: FontWeight.bold),
                      ),
                      TextButton(
                        onPressed: () => _launchUrl(
                          'https://huggingface.co/unsloth/gemma-4-2b-it-GGUF',
                        ),
                        child: const Text('Gemma 4 E2B IT (Unsloth GGUF Repo)'),
                      ),
                      const Divider(),
                      if (_localModels.isNotEmpty) ...[
                        const Text(
                          'Local models found in Documents / models',
                          style: TextStyle(fontWeight: FontWeight.bold),
                        ),
                        ..._localModels.map(
                          (file) => ListTile(
                            title: Text(file.path.split('/').last),
                            leading: const Icon(Icons.model_training),
                            selected: _modelPath == file.path,
                            onTap: _isGenerating
                                ? null
                                : () => _selectModel(file.path),
                          ),
                        ),
                        const Divider(),
                      ],
                      ElevatedButton(
                        onPressed: _isGenerating ? null : _pickModel,
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.blue,
                          foregroundColor: Colors.white,
                        ),
                        child: Text(
                          _modelPath == null
                              ? 'SELECT LOCAL GGUF MODEL'
                              : 'Model: ${_modelPath!.split('/').last}',
                        ),
                      ),
                      const SizedBox(height: 16),
                      DropdownButtonFormField<String>(
                        initialValue:
                            availableCacheTypes.contains(_selectedCacheType)
                            ? _selectedCacheType
                            : 'q8_0',
                        decoration: const InputDecoration(
                          labelText: 'KV-cache V type',
                          border: OutlineInputBorder(),
                        ),
                        items: availableCacheTypes.map((cacheType) {
                          final option = _kvOptions[cacheType]!;
                          return DropdownMenuItem<String>(
                            value: cacheType,
                            child: Text(option.dropdownLabel),
                          );
                        }).toList(),
                        onChanged: _isGenerating || _isValidating
                            ? null
                            : (value) {
                                if (value == null) {
                                  return;
                                }
                                setState(() {
                                  _selectedCacheType = value;
                                  _cacheTypeChosenByUser = true;
                                });
                              },
                      ),
                      const SizedBox(height: 12),
                      Text(
                        _effectiveRuntimeDescription(),
                        style: const TextStyle(fontStyle: FontStyle.italic),
                      ),
                      const SizedBox(height: 16),
                      _buildValidationSection(),
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
                              onPressed:
                                  _isGenerating ||
                                      _isValidating ||
                                      _modelPath == null
                                  ? null
                                  : _generate,
                              child: _isGenerating
                                  ? const SizedBox(
                                      height: 20,
                                      width: 20,
                                      child: CircularProgressIndicator(
                                        strokeWidth: 2,
                                      ),
                                    )
                                  : const Text('Generate'),
                            ),
                          ),
                          if (_isGenerating) ...[
                            const SizedBox(width: 8),
                            ElevatedButton(
                              onPressed: _stop,
                              style: ElevatedButton.styleFrom(
                                backgroundColor: Colors.red,
                                foregroundColor: Colors.white,
                              ),
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
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 8),
              const Text(
                'Response:',
                style: TextStyle(fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 4),
              SizedBox(
                height: 220,
                child: Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    border: Border.all(color: Colors.grey),
                    borderRadius: BorderRadius.circular(4),
                  ),
                  child: SingleChildScrollView(child: Text(_response)),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
