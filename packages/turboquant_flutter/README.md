# turboquant_flutter

A new Flutter plugin project.

## Getting Started

This project is a starting point for a Flutter
[plug-in package](https://flutter.dev/to/develop-plugins),
a specialized package that includes platform-specific implementation code for
Android and/or iOS.

For help getting started with Flutter development, view the
[online documentation](https://docs.flutter.dev), which offers tutorials,
samples, guidance on mobile development, and a full API reference.

## iOS Native Artifacts

The iOS plugin consumes prebuilt static archives from `ios/libs/`, and that
directory is intentionally gitignored. After rebuilding the native wrapper, sync
the generated archives and public header into the plugin before running the iOS
example:

```bash
./scripts/sync_ios_plugin_artifacts.sh device
```

Use `simulator` instead of `device` when you want the plugin wired to the
`iphonesimulator` build outputs. The example app now validates TurboQuant
`turbo3` and `turbo4` paths at runtime and will expose CPU-KV fallback when
Metal KV offload is not valid for the current device/model combination.
