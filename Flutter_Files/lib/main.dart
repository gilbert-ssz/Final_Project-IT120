import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_v2/tflite_v2.dart';
import 'package:fl_chart/fl_chart.dart';

import 'dart:developer' as devtools;
import 'firebase_options.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp(
    options: DefaultFirebaseOptions.currentPlatform,
  );
  runApp(const MyApp());
}

// Detection History Item Model
class DetectionItem {
  final String imagePath;
  final String label;
  final double confidence;
  final DateTime timestamp;
  final bool isUnknown;
  final Map<String, double> allPredictions; // All class predictions

  DetectionItem({
    required this.imagePath,
    required this.label,
    required this.confidence,
    required this.timestamp,
    this.isUnknown = false,
    this.allPredictions = const {},
  });
}

// Global list para sa history
List<DetectionItem> detectionHistory = [];

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Root Goods Detection',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF2E5F4F),
          brightness: Brightness.light,
        ),
        scaffoldBackgroundColor: const Color(0xFFF8F9F7),
        useMaterial3: true,
        fontFamily: 'Inter',
        appBarTheme: const AppBarTheme(
          backgroundColor: Color(0xFFF8F9F7),
          elevation: 0,
          centerTitle: false,
        ),
      ),
      home: const MainScreen(),
    );
  }
}

class MainScreen extends StatefulWidget {
  const MainScreen({super.key});

  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> {
  int _currentIndex = 0;

  void _onHistoryUpdate() {
    setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    final List<Widget> screens = [
      MyHomePage(onHistoryUpdate: _onHistoryUpdate),
      AnalyticsPage(key: ValueKey(detectionHistory.length)),
      const AboutPage(),
    ];

    return Scaffold(
      body: screens[_currentIndex],
      bottomNavigationBar: Container(
        decoration: BoxDecoration(
          color: Colors.white,
          border: Border(
            top: BorderSide(
              color: Colors.grey.shade200,
              width: 0.5,
            ),
          ),
        ),
        child: SafeArea(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                _buildNavItem(0, Icons.home_rounded, 'Detect'),
                _buildNavItem(1, Icons.bar_chart_rounded, 'Analytics'),
                _buildNavItem(2, Icons.info_outline_rounded, 'About'),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildNavItem(int index, IconData icon, String label) {
    final isSelected = _currentIndex == index;
    return GestureDetector(
      onTap: () => setState(() => _currentIndex = index),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding: EdgeInsets.symmetric(
          horizontal: isSelected ? 20 : 12,
          vertical: 10,
        ),
        decoration: BoxDecoration(
          color: isSelected ? const Color(0xFF1A1A1A) : Colors.transparent,
          borderRadius: BorderRadius.circular(20),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              icon,
              color: isSelected ? Colors.white : Colors.grey.shade600,
              size: 22,
            ),
            if (isSelected) ...[
              const SizedBox(width: 8),
              Text(
                label,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 13,
                  fontWeight: FontWeight.w600,
                  letterSpacing: 0.3,
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

class MyHomePage extends StatefulWidget {
  final VoidCallback onHistoryUpdate;

  const MyHomePage({super.key, required this.onHistoryUpdate});

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  File? filePath;
  String label = '';
  double confidence = 0.0;
  bool isLoading = false;
  bool isUnknown = false;
  Map<String, double> currentPredictions =
      {}; // Store current predictions for display

  Future<void> _tfLteInit() async {
    try {
      String? res = await Tflite.loadModel(
          model: "assets/model_unquant.tflite",
          labels: "assets/labels.txt",
          numThreads: 1,
          isAsset: true,
          useGpuDelegate: false);
      devtools.log("Model loaded: $res");
    } catch (e) {
      devtools.log("Error loading model: $e");
    }
  }

  Future<void> _addToHistoryAndFirestore(String imagePath, String detectedLabel,
      double conf, bool unknown, Map<String, double> allPreds) async {
    detectionHistory.insert(
      0,
      DetectionItem(
        imagePath: imagePath,
        label: detectedLabel,
        confidence: conf,
        timestamp: DateTime.now(),
        allPredictions: allPreds,
      ),
    );

    widget.onHistoryUpdate();

    try {
      await FirebaseFirestore.instance.collection('detection_history').add({
        'Accuracy_rate': conf,
        'Class_type': detectedLabel,
        'Time': Timestamp.now(),
      });
      devtools.log("✅ Detection saved to Firestore: $detectedLabel ($conf%)");
    } catch (e) {
      devtools.log("❌ Failed to save detection to Firestore: $e");
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Failed to save to database: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  Future<void> pickImageGallery() async {
    try {
      final ImagePicker picker = ImagePicker();
      final XFile? image = await picker.pickImage(source: ImageSource.gallery);

      if (image == null) return;

      var imageMap = File(image.path);

      setState(() {
        filePath = imageMap;
        isLoading = true;
        label = '';
        confidence = 0.0;
        isUnknown = false;
      });

      var recognitions = await Tflite.runModelOnImage(
          path: image.path,
          imageMean: 0.0,
          imageStd: 255.0,
          numResults: 10,
          threshold: 0.0,
          asynch: true);

      if (recognitions == null || recognitions.isEmpty) {
        devtools.log("recognitions is Null or Empty");
        setState(() {
          isLoading = false;
          label = 'Kani na image wala sa akong class';
          isUnknown = true;
          confidence = 0.0;
        });
        await _addToHistoryAndFirestore(
            image.path, 'Unknown - Not in Class', 0.0, true, {});
        return;
      }

      devtools.log(recognitions.toString());

      // Build all predictions map
      Map<String, double> allPreds = {};
      for (var rec in recognitions) {
        String predLabel = rec['label'].toString();
        double predConf = (rec['confidence'] as num).toDouble() * 100;
        allPreds[predLabel] = predConf;
      }

      double detectedConfidence = (recognitions[0]['confidence'] * 100);
      if (detectedConfidence < 30) {
        setState(() {
          confidence = detectedConfidence;
          label = 'Kani na image walla sa akong class';
          isUnknown = true;
          isLoading = false;
          currentPredictions = allPreds; // Store predictions
        });
        await _addToHistoryAndFirestore(image.path, 'Unknown - Low Confidence',
            detectedConfidence, true, allPreds);
      } else {
        String detectedLabel = recognitions[0]['label'].toString();
        setState(() {
          confidence = detectedConfidence;
          label = detectedLabel;
          isUnknown = false;
          isLoading = false;
          currentPredictions = allPreds; // Store predictions
        });
        await _addToHistoryAndFirestore(
            image.path, detectedLabel, detectedConfidence, false, allPreds);
      }
    } catch (e) {
      devtools.log("Error in pickImageGallery: $e");
      setState(() {
        isLoading = false;
      });
    }
  }

  Future<void> pickImageCamera() async {
    try {
      final ImagePicker picker = ImagePicker();
      final XFile? image = await picker.pickImage(source: ImageSource.camera);

      if (image == null) return;

      var imageMap = File(image.path);

      setState(() {
        filePath = imageMap;
        isLoading = true;
        label = '';
        confidence = 0.0;
        isUnknown = false;
      });

      var recognitions = await Tflite.runModelOnImage(
          path: image.path,
          imageMean: 0.0,
          imageStd: 255.0,
          numResults: 10,
          threshold: 0.0,
          asynch: true);

      if (recognitions == null || recognitions.isEmpty) {
        devtools.log("recognitions is Null or Empty");
        setState(() {
          isLoading = false;
          label = 'Kani na image wala sa akong class';
          isUnknown = true;
          confidence = 0.0;
        });
        await _addToHistoryAndFirestore(
            image.path, 'Unknown - Not in Class', 0.0, true, {});
        return;
      }

      devtools.log(recognitions.toString());

      // Build all predictions map
      Map<String, double> allPreds = {};
      for (var rec in recognitions) {
        String predLabel = rec['label'].toString();
        double predConf = (rec['confidence'] as num).toDouble() * 100;
        allPreds[predLabel] = predConf;
      }

      double detectedConfidence = (recognitions[0]['confidence'] * 100);
      if (detectedConfidence < 30) {
        setState(() {
          confidence = detectedConfidence;
          label = 'Kani na image wala sa akong class';
          isUnknown = true;
          isLoading = false;
          currentPredictions = allPreds; // Store predictions
        });
        await _addToHistoryAndFirestore(image.path, 'Unknown - Low Confidence',
            detectedConfidence, true, allPreds);
      } else {
        String detectedLabel = recognitions[0]['label'].toString();
        setState(() {
          confidence = detectedConfidence;
          label = detectedLabel;
          isUnknown = false;
          isLoading = false;
          currentPredictions = allPreds; // Store predictions
        });
        await _addToHistoryAndFirestore(
            image.path, detectedLabel, detectedConfidence, false, allPreds);
      }
    } catch (e) {
      devtools.log("Error in pickImageCamera: $e");
      setState(() {
        isLoading = false;
      });
    }
  }

  @override
  void dispose() {
    Tflite.close();
    super.dispose();
  }

  @override
  void initState() {
    super.initState();
    _tfLteInit();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        constraints: BoxConstraints(
          minHeight: MediaQuery.of(context).size.height, // 👈 IMPORTANT
        ),
        decoration: const BoxDecoration(
          image: DecorationImage(
            image: AssetImage(
              'assets/Midjourney_  Vibrant rural landscape with sustainable farming and renewable energy_.jfif',
            ),
            fit: BoxFit.cover,
          ),
        ),
        child: SafeArea(
          child: SingleChildScrollView(
            child: Padding(
              padding: const EdgeInsets.all(24.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const SizedBox(height: 20),
                  // Header with enhanced styling
                  Container(
                    decoration: BoxDecoration(
                      gradient: LinearGradient(
                        colors: [
                          const Color(0xFF2E5F4F).withOpacity(0.08),
                          const Color(0xFF1A3A32).withOpacity(0.04),
                        ],
                        begin: Alignment.topLeft,
                        end: Alignment.bottomRight,
                      ),
                      borderRadius: BorderRadius.circular(20),
                      border: Border.all(
                        color: const Color(0xFF2E5F4F).withOpacity(0.1),
                        width: 1,
                      ),
                    ),
                    padding: const EdgeInsets.all(16),
                    child: Row(
                      children: [
                        Container(
                          padding: const EdgeInsets.all(12),
                          decoration: BoxDecoration(
                            gradient: LinearGradient(
                              colors: [
                                const Color(0xFF2E5F4F).withOpacity(0.9),
                                const Color(0xFF1A3A32),
                              ],
                              begin: Alignment.topLeft,
                              end: Alignment.bottomRight,
                            ),
                            borderRadius: BorderRadius.circular(16),
                            boxShadow: [
                              BoxShadow(
                                color:
                                    const Color(0xFF2E5F4F).withOpacity(0.15),
                                blurRadius: 12,
                                offset: const Offset(0, 4),
                              ),
                            ],
                          ),
                          child: const Icon(
                            Icons.spa_outlined,
                            size: 28,
                            color: Colors.white,
                          ),
                        ),
                        const SizedBox(width: 16),
                        const Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                'Root Goods',
                                style: TextStyle(
                                  fontSize: 24,
                                  fontWeight: FontWeight.w700,
                                  color: Color(0xFF1A1A1A),
                                  letterSpacing: -0.5,
                                ),
                              ),
                              SizedBox(height: 2),
                              Text(
                                'Advanced Root Crop Detection',
                                style: TextStyle(
                                  fontSize: 13,
                                  color: Color(0xFF666666),
                                  fontWeight: FontWeight.w400,
                                  letterSpacing: 0.2,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 32),

                  // Image Container
                  Container(
                    decoration: BoxDecoration(
                      gradient: LinearGradient(
                        colors: [
                          Colors.white,
                          const Color(0xFFF8FFF8).withOpacity(0.5),
                        ],
                        begin: Alignment.topLeft,
                        end: Alignment.bottomRight,
                      ),
                      borderRadius: BorderRadius.circular(25),
                      border: Border.all(
                        color: const Color(0xFF2E5F4F).withOpacity(0.1),
                        width: 1,
                      ),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.black.withOpacity(0.08),
                          blurRadius: 16,
                          offset: const Offset(0, 4),
                        ),
                      ],
                    ),
                    child: Column(
                      children: [
                        Container(
                          height: 360,
                          decoration: const BoxDecoration(
                            color: Color(0xFFF5F5F5),
                            borderRadius: BorderRadius.vertical(
                              top: Radius.circular(24),
                            ),
                          ),
                          child: ClipRRect(
                            borderRadius: const BorderRadius.vertical(
                              top: Radius.circular(23),
                            ),
                            child: filePath == null
                                ? Center(
                                    child: Column(
                                      mainAxisAlignment:
                                          MainAxisAlignment.center,
                                      children: [
                                        Container(
                                          padding: const EdgeInsets.all(24),
                                          decoration: BoxDecoration(
                                            color: const Color(0xFFF8F8F8),
                                            shape: BoxShape.circle,
                                            border: Border.all(
                                              color: Colors.grey.shade200,
                                              width: 2,
                                            ),
                                          ),
                                          child: Icon(
                                            Icons.image_outlined,
                                            size: 48,
                                            color: Colors.grey.shade400,
                                          ),
                                        ),
                                        const SizedBox(height: 20),
                                        const Text(
                                          'No Image Selected',
                                          style: TextStyle(
                                            fontSize: 16,
                                            fontWeight: FontWeight.w600,
                                            color: Color(0xFF1A1A1A),
                                            letterSpacing: 0.2,
                                          ),
                                        ),
                                        const SizedBox(height: 6),
                                        Text(
                                          'Upload an image to begin analysis',
                                          style: TextStyle(
                                            fontSize: 13,
                                            color: const Color.fromARGB(255, 0, 0, 0),
                                            letterSpacing: 0.1,
                                          ),
                                        ),
                                      ],
                                    ),
                                  )
                                : Stack(
                                    fit: StackFit.expand,
                                    children: [
                                      Image.file(
                                        filePath!,
                                        fit: BoxFit.cover,
                                      ),
                                      if (isLoading)
                                        Container(
                                          color: Colors.white.withOpacity(0.95),
                                          child: Center(
                                            child: Column(
                                              mainAxisAlignment:
                                                  MainAxisAlignment.center,
                                              children: [
                                                Container(
                                                  width: 48,
                                                  height: 48,
                                                  padding:
                                                      const EdgeInsets.all(12),
                                                  decoration: BoxDecoration(
                                                    color:
                                                        const Color(0xFF1A1A1A),
                                                    borderRadius:
                                                        BorderRadius.circular(
                                                            12),
                                                  ),
                                                  child:
                                                      const CircularProgressIndicator(
                                                    strokeWidth: 2.5,
                                                    valueColor:
                                                        AlwaysStoppedAnimation<
                                                                Color>(
                                                            Colors.white),
                                                  ),
                                                ),
                                                const SizedBox(height: 16),
                                                const Text(
                                                  'Analyzing image...',
                                                  style: TextStyle(
                                                    fontSize: 14,
                                                    fontWeight: FontWeight.w500,
                                                    color: Color(0xFF1A1A1A),
                                                    letterSpacing: 0.2,
                                                  ),
                                                ),
                                              ],
                                            ),
                                          ),
                                        ),
                                    ],
                                  ),
                          ),
                        ),

                        // Results Section
                        if (label.isNotEmpty && !isLoading)
                          Container(
                            width: double.infinity,
                            padding: const EdgeInsets.all(24),
                            decoration: BoxDecoration(
                              color: isUnknown
                                  ? const Color(0xFFFFF5F5)
                                  : const Color(0xFFF8FFF8),
                              borderRadius: const BorderRadius.vertical(
                                bottom: Radius.circular(24),
                              ),
                            ),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Row(
                                  children: [
                                    Container(
                                      padding: const EdgeInsets.all(8),
                                      decoration: BoxDecoration(
                                        color: isUnknown
                                            ? const Color(0xFFFFE5E5)
                                            : const Color(0xFFE8F5E9),
                                        borderRadius: BorderRadius.circular(10),
                                      ),
                                      child: Icon(
                                        isUnknown
                                            ? Icons.warning_amber_rounded
                                            : Icons.check_circle_rounded,
                                        color: isUnknown
                                            ? const Color(0xFFD32F2F)
                                            : const Color(0xFF2E7D32),
                                        size: 20,
                                      ),
                                    ),
                                    const SizedBox(width: 12),
                                    Text(
                                      isUnknown
                                          ? 'Detection Failed'
                                          : 'Detected',
                                      style: TextStyle(
                                        fontSize: 12,
                                        fontWeight: FontWeight.w600,
                                        color: isUnknown
                                            ? const Color(0xFFD32F2F)
                                            : const Color(0xFF2E7D32),
                                        letterSpacing: 1.2,
                                      ),
                                    ),
                                  ],
                                ),
                                const SizedBox(height: 16),
                                Text(
                                  label,
                                  style: const TextStyle(
                                    fontSize: 22,
                                    fontWeight: FontWeight.w700,
                                    color: Color(0xFF1A1A1A),
                                    letterSpacing: -0.3,
                                    height: 1.3,
                                  ),
                                ),
                                if (confidence > 0) ...[
                                  const SizedBox(height: 16),
                                  Row(
                                    children: [
                                      Expanded(
                                        child: Column(
                                          crossAxisAlignment:
                                              CrossAxisAlignment.start,
                                          children: [
                                            Text(
                                              'Confidence Score',
                                              style: TextStyle(
                                                fontSize: 11,
                                                fontWeight: FontWeight.w600,
                                                color: Colors.grey.shade600,
                                                letterSpacing: 0.5,
                                              ),
                                            ),
                                            const SizedBox(height: 8),
                                            ClipRRect(
                                              borderRadius:
                                                  BorderRadius.circular(8),
                                              child: LinearProgressIndicator(
                                                value: confidence / 100,
                                                backgroundColor: isUnknown
                                                    ? Colors.red.shade100
                                                    : Colors.grey.shade200,
                                                valueColor:
                                                    AlwaysStoppedAnimation<
                                                        Color>(
                                                  isUnknown
                                                      ? const Color(0xFFD32F2F)
                                                      : const Color(0xFF1A1A1A),
                                                ),
                                                minHeight: 8,
                                              ),
                                            ),
                                          ],
                                        ),
                                      ),
                                      const SizedBox(width: 16),
                                      Container(
                                        padding: const EdgeInsets.symmetric(
                                          horizontal: 16,
                                          vertical: 8,
                                        ),
                                        decoration: BoxDecoration(
                                          color: isUnknown
                                              ? const Color(0xFFFFE5E5)
                                              : const Color(0xFF1A1A1A),
                                          borderRadius:
                                              BorderRadius.circular(12),
                                        ),
                                        child: Text(
                                          '${confidence.toStringAsFixed(1)}%',
                                          style: TextStyle(
                                            fontSize: 16,
                                            fontWeight: FontWeight.w700,
                                            color: isUnknown
                                                ? const Color(0xFFD32F2F)
                                                : Colors.white,
                                            letterSpacing: 0.3,
                                          ),
                                        ),
                                      ),
                                    ],
                                  ),
                                ],
                              ],
                            ),
                          ),
                        // Prediction Distribution Section
                        if (currentPredictions.isNotEmpty && label.isNotEmpty)
                          Container(
                            width: double.infinity,
                            padding: const EdgeInsets.all(24),
                            decoration: const BoxDecoration(
                              color: Color(0xFFF5F5F5),
                              borderRadius: BorderRadius.vertical(
                                bottom: Radius.circular(24),
                              ),
                            ),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Row(
                                  mainAxisAlignment:
                                      MainAxisAlignment.spaceBetween,
                                  children: [
                                    const Text(
                                      'Prediction Distribution',
                                      style: TextStyle(
                                        fontSize: 14,
                                        fontWeight: FontWeight.w700,
                                        color: Color(0xFF1A1A1A),
                                      ),
                                    ),
                                    Container(
                                      padding: const EdgeInsets.symmetric(
                                          horizontal: 12, vertical: 4),
                                      decoration: BoxDecoration(
                                        color: const Color(0xFFFFF5F5),
                                        borderRadius: BorderRadius.circular(8),
                                      ),
                                      child: Text(
                                        'Top: $label',
                                        style: const TextStyle(
                                          fontSize: 11,
                                          fontWeight: FontWeight.w600,
                                          color: Color(0xFFD32F2F),
                                        ),
                                      ),
                                    ),
                                  ],
                                ),
                                const SizedBox(height: 16),
                                ...currentPredictions.entries
                                    .toList()
                                    .asMap()
                                    .entries
                                    .map((entry) {
                                  int index = entry.key;
                                  String className = entry.value.key;
                                  double conf = entry.value.value;

                                  return Padding(
                                    padding: const EdgeInsets.only(bottom: 12),
                                    child: Column(
                                      crossAxisAlignment:
                                          CrossAxisAlignment.start,
                                      children: [
                                        Row(
                                          mainAxisAlignment:
                                              MainAxisAlignment.spaceBetween,
                                          children: [
                                            Text(
                                              className,
                                              style: const TextStyle(
                                                fontSize: 12,
                                                fontWeight: FontWeight.w600,
                                                color: Color(0xFF1A1A1A),
                                              ),
                                            ),
                                            Text(
                                              '${conf.toStringAsFixed(2)}%',
                                              style: const TextStyle(
                                                fontSize: 12,
                                                fontWeight: FontWeight.w700,
                                                color: Color(0xFF1A1A1A),
                                              ),
                                            ),
                                          ],
                                        ),
                                        const SizedBox(height: 4),
                                        ClipRRect(
                                          borderRadius:
                                              BorderRadius.circular(4),
                                          child: LinearProgressIndicator(
                                            value: conf / 100,
                                            backgroundColor:
                                                Colors.grey.shade300,
                                            valueColor:
                                                AlwaysStoppedAnimation<Color>(
                                              _getPredictionColor(index),
                                            ),
                                            minHeight: 6,
                                          ),
                                        ),
                                      ],
                                    ),
                                  );
                                }).toList(),
                              ],
                            ),
                          ),
                      ],
                    ),
                  ),

                  const SizedBox(height: 24),

                  // Action Buttons
                  Row(
                    children: [
                      Expanded(
                        child: _buildActionButton(
                          onPressed: pickImageCamera,
                          icon: Icons.camera_alt_outlined,
                          label: 'Camera',
                          isPrimary: true,
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: _buildActionButton(
                          onPressed: pickImageGallery,
                          icon: Icons.photo_library_outlined,
                          label: 'Gallery',
                          isPrimary: false,
                        ),
                      ),
                    ],
                  ),

                  const SizedBox(height: 24),

                  // Info Card
                  Container(
                    padding: const EdgeInsets.all(20),
                    decoration: BoxDecoration(
                      gradient: LinearGradient(
                        colors: [
                          const Color(0xFFFFFBF5).withOpacity(0.8),
                          const Color(0xFFFFF8F0),
                        ],
                        begin: Alignment.topLeft,
                        end: Alignment.bottomRight,
                      ),
                      borderRadius: BorderRadius.circular(20),
                      border: Border.all(
                        color: const Color(0xFFFFE0B2),
                        width: 1,
                      ),
                    ),
                    child: Row(
                      children: [
                        Container(
                          padding: const EdgeInsets.all(10),
                          decoration: BoxDecoration(
                            color: const Color(0xFFFFECB3),
                            borderRadius: BorderRadius.circular(12),
                          ),
                          child: const Icon(
                            Icons.tips_and_updates_outlined,
                            color: Color(0xFFE65100),
                            size: 20,
                          ),
                        ),
                        const SizedBox(width: 16),
                        const Expanded(
                          child: Text(
                            'Ensure proper lighting and focus for accurate detection',
                            style: TextStyle(
                              fontSize: 13,
                              color: Color(0xFF5D4037),
                              fontWeight: FontWeight.w500,
                              height: 1.4,
                              letterSpacing: 0.1,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  Color _getPredictionColor(int index) {
    final colors = [
      const Color(0xFFFF6B6B),
      const Color(0xFF4ECDC4),
      const Color(0xFF45B7D1),
      const Color(0xFFFFA07A),
      const Color(0xFF98D8C8),
      const Color(0xFFF7DC6F),
      const Color(0xFFBB8FCE),
      const Color(0xFF85C1E2),
      const Color(0xFFF8B88B),
      const Color(0xFF82E0AA),
    ];
    return colors[index % colors.length];
  }

  Widget _buildActionButton({
    required VoidCallback onPressed,
    required IconData icon,
    required String label,
    required bool isPrimary,
  }) {
    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: onPressed,
        borderRadius: BorderRadius.circular(16),
        child: Container(
          padding: const EdgeInsets.symmetric(vertical: 18),
          decoration: BoxDecoration(
            gradient: isPrimary
                ? LinearGradient(
                    colors: [
                      const Color(0xFF2E5F4F).withOpacity(0.9),
                      const Color(0xFF1A3A32),
                    ],
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                  )
                : null,
            color: !isPrimary ? Colors.white : null,
            borderRadius: BorderRadius.circular(16),
            border: isPrimary
                ? null
                : Border.all(
                    color: const Color(0xFF2E5F4F).withOpacity(0.2),
                    width: 1.5),
            boxShadow: isPrimary
                ? [
                    BoxShadow(
                      color: const Color(0xFF2E5F4F).withOpacity(0.2),
                      blurRadius: 12,
                      offset: const Offset(0, 4),
                    ),
                  ]
                : null,
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(
                icon,
                color: isPrimary ? Colors.white : const Color(0xFF2E5F4F),
                size: 20,
              ),
              const SizedBox(width: 10),
              Text(
                label,
                style: TextStyle(
                  fontSize: 15,
                  fontWeight: FontWeight.w600,
                  color: isPrimary ? Colors.white : const Color(0xFF2E5F4F),
                  letterSpacing: 0.3,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class HistoryPage extends StatefulWidget {
  const HistoryPage({super.key});

  @override
  State<HistoryPage> createState() => _HistoryPageState();
}

class _HistoryPageState extends State<HistoryPage> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFFAFAFA),
      body: SafeArea(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Padding(
              padding: const EdgeInsets.all(24.0),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  const Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Detection History',
                        style: TextStyle(
                          fontSize: 24,
                          fontWeight: FontWeight.w700,
                          color: Color(0xFF1A1A1A),
                          letterSpacing: -0.5,
                        ),
                      ),
                      SizedBox(height: 4),
                      Text(
                        'Your past scans and results',
                        style: TextStyle(
                          fontSize: 13,
                          color: Color(0xFF666666),
                          fontWeight: FontWeight.w400,
                        ),
                      ),
                    ],
                  ),
                  if (detectionHistory.isNotEmpty)
                    IconButton(
                      icon: Container(
                        padding: const EdgeInsets.all(8),
                        decoration: BoxDecoration(
                          color: Colors.white,
                          borderRadius: BorderRadius.circular(12),
                          border: Border.all(color: Colors.grey.shade200),
                        ),
                        child: const Icon(
                          Icons.delete_outline_rounded,
                          size: 20,
                        ),
                      ),
                      onPressed: () {
                        showDialog(
                          context: context,
                          builder: (context) => AlertDialog(
                            backgroundColor: Colors.white,
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(20),
                            ),
                            title: const Text(
                              'Clear History',
                              style: TextStyle(
                                fontWeight: FontWeight.w700,
                                fontSize: 20,
                              ),
                            ),
                            content: const Text(
                              'Are you sure you want to clear all detection history?',
                              style: TextStyle(
                                fontSize: 14,
                                color: Color(0xFF666666),
                              ),
                            ),
                            actions: [
                              TextButton(
                                onPressed: () => Navigator.pop(context),
                                child: const Text(
                                  'Cancel',
                                  style: TextStyle(
                                    color: Color(0xFF666666),
                                    fontWeight: FontWeight.w600,
                                  ),
                                ),
                              ),
                              TextButton(
                                onPressed: () {
                                  setState(() {
                                    detectionHistory.clear();
                                  });
                                  Navigator.pop(context);
                                },
                                child: const Text(
                                  'Clear All',
                                  style: TextStyle(
                                    color: Color(0xFFD32F2F),
                                    fontWeight: FontWeight.w600,
                                  ),
                                ),
                              ),
                            ],
                          ),
                        );
                      },
                    ),
                ],
              ),
            ),
            Expanded(
              child: detectionHistory.isEmpty
                  ? Center(
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Container(
                            padding: const EdgeInsets.all(32),
                            decoration: BoxDecoration(
                              color: const Color(0xFFF5F5F5),
                              shape: BoxShape.circle,
                              border: Border.all(
                                  color: Colors.grey.shade200, width: 2),
                            ),
                            child: Icon(
                              Icons.analytics_outlined,
                              size: 64,
                              color: Colors.grey.shade400,
                            ),
                          ),
                          const SizedBox(height: 24),
                          const Text(
                            'No History Yet',
                            style: TextStyle(
                              fontSize: 20,
                              fontWeight: FontWeight.w700,
                              color: Color(0xFF1A1A1A),
                            ),
                          ),
                          const SizedBox(height: 8),
                          Text(
                            'Your detection history will appear here',
                            style: TextStyle(
                              fontSize: 14,
                              color: Colors.grey.shade600,
                            ),
                          ),
                        ],
                      ),
                    )
                  : ListView.builder(
                      padding: const EdgeInsets.symmetric(horizontal: 24),
                      itemCount: detectionHistory.length,
                      itemBuilder: (context, index) {
                        final item = detectionHistory[index];
                        return Padding(
                          padding: const EdgeInsets.only(bottom: 16.0),
                          child: Container(
                            decoration: BoxDecoration(
                              color: Colors.white,
                              borderRadius: BorderRadius.circular(20),
                              border: Border.all(
                                  color: Colors.grey.shade200, width: 1),
                            ),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                ClipRRect(
                                  borderRadius: const BorderRadius.vertical(
                                    top: Radius.circular(20),
                                  ),
                                  child: Image.file(
                                    File(item.imagePath),
                                    height: 180,
                                    width: double.infinity,
                                    fit: BoxFit.cover,
                                    errorBuilder: (context, error, stackTrace) {
                                      return Container(
                                        height: 180,
                                        color: Colors.grey.shade200,
                                        child: Icon(
                                          Icons.broken_image_outlined,
                                          size: 48,
                                          color: Colors.grey.shade400,
                                        ),
                                      );
                                    },
                                  ),
                                ),
                                Padding(
                                  padding: const EdgeInsets.all(16.0),
                                  child: Column(
                                    crossAxisAlignment:
                                        CrossAxisAlignment.start,
                                    children: [
                                      Row(
                                        children: [
                                          Container(
                                            padding: const EdgeInsets.all(6),
                                            decoration: BoxDecoration(
                                              color: item.isUnknown
                                                  ? const Color(0xFFFFE5E5)
                                                  : const Color(0xFFE8F5E9),
                                              borderRadius:
                                                  BorderRadius.circular(8),
                                            ),
                                            child: Icon(
                                              item.isUnknown
                                                  ? Icons.warning_amber_rounded
                                                  : Icons.check_circle_rounded,
                                              color: item.isUnknown
                                                  ? const Color(0xFFD32F2F)
                                                  : const Color(0xFF2E7D32),
                                              size: 16,
                                            ),
                                          ),
                                          const SizedBox(width: 12),
                                          Expanded(
                                            child: Text(
                                              item.label,
                                              style: TextStyle(
                                                fontSize: 16,
                                                fontWeight: FontWeight.w700,
                                                color: item.isUnknown
                                                    ? const Color(0xFFD32F2F)
                                                    : const Color(0xFF1A1A1A),
                                                letterSpacing: -0.2,
                                              ),
                                            ),
                                          ),
                                        ],
                                      ),
                                      const SizedBox(height: 12),
                                      Row(
                                        mainAxisAlignment:
                                            MainAxisAlignment.spaceBetween,
                                        children: [
                                          if (item.confidence > 0)
                                            Container(
                                              padding:
                                                  const EdgeInsets.symmetric(
                                                horizontal: 12,
                                                vertical: 6,
                                              ),
                                              decoration: BoxDecoration(
                                                color: item.isUnknown
                                                    ? const Color(0xFFFFF5F5)
                                                    : const Color(0xFFF5F5F5),
                                                borderRadius:
                                                    BorderRadius.circular(8),
                                              ),
                                              child: Text(
                                                '${item.confidence.toStringAsFixed(1)}%',
                                                style: TextStyle(
                                                  fontSize: 13,
                                                  fontWeight: FontWeight.w600,
                                                  color: item.isUnknown
                                                      ? const Color(0xFFD32F2F)
                                                      : const Color(0xFF1A1A1A),
                                                ),
                                              ),
                                            ),
                                          Row(
                                            children: [
                                              Icon(
                                                Icons.access_time_rounded,
                                                size: 14,
                                                color: Colors.grey.shade600,
                                              ),
                                              const SizedBox(width: 4),
                                              Text(
                                                _formatDateTime(item.timestamp),
                                                style: TextStyle(
                                                  fontSize: 12,
                                                  color: Colors.grey.shade600,
                                                ),
                                              ),
                                            ],
                                          ),
                                        ],
                                      ),
                                    ],
                                  ),
                                ),
                              ],
                            ),
                          ),
                        );
                      },
                    ),
            ),
          ],
        ),
      ),
    );
  }

  String _formatDateTime(DateTime dateTime) {
    final now = DateTime.now();
    final difference = now.difference(dateTime);
    if (difference.inMinutes < 1) {
      return 'Just now';
    } else if (difference.inHours < 1) {
      return '${difference.inMinutes}m ago';
    } else if (difference.inDays < 1) {
      return '${difference.inHours}h ago';
    } else if (difference.inDays == 1) {
      return 'Yesterday';
    } else {
      return '${dateTime.day}/${dateTime.month}/${dateTime.year}';
    }
  }
}

class AnalyticsPage extends StatefulWidget {
  const AnalyticsPage({super.key});

  @override
  State<AnalyticsPage> createState() => _AnalyticsPageState();
}

class _AnalyticsPageState extends State<AnalyticsPage> {
  // List of all root vegetable classes from your model
  final List<String> allClasses = [
    'potato',
    'gabi',
    'luya',
    'carrots',
    'singkamas',
    'garlic',
    'onion',
    'kamote',
    'ubi',
    'balanghoy',
  ];

  // Calculate detection count per class
  Map<String, int> _getDetectionCountPerClass() {
    Map<String, int> counts = {};

    // Initialize all classes with 0
    for (String className in allClasses) {
      counts[className] = 0;
    }

    // Count detections for each class
    for (DetectionItem item in detectionHistory) {
      if (!item.isUnknown) {
        // Extract class name from label
        String? className;
        for (String cls in allClasses) {
          if (item.label.toLowerCase().contains(cls.toLowerCase())) {
            className = cls;
            break;
          }
        }

        if (className != null) {
          counts[className] = (counts[className] ?? 0) + 1;
        }
      }
    }

    return counts;
  }

  // Calculate average confidence per class
  Map<String, double> _getAverageConfidencePerClass() {
    Map<String, List<double>> confidences = {};

    // Initialize all classes
    for (String className in allClasses) {
      confidences[className] = [];
    }

    // Collect confidences for each class from all predictions
    for (DetectionItem item in detectionHistory) {
      if (!item.isUnknown && item.allPredictions.isNotEmpty) {
        for (String className in allClasses) {
          if (item.allPredictions.containsKey(className)) {
            confidences[className]!.add(item.allPredictions[className]!);
          }
        }
      }
    }

    // Calculate averages
    Map<String, double> averages = {};
    for (String className in allClasses) {
      if (confidences[className]!.isNotEmpty) {
        double avg = confidences[className]!.reduce((a, b) => a + b) /
            confidences[className]!.length;
        averages[className] = avg;
      } else {
        averages[className] = 0.0;
      }
    }

    return averages;
  }

  // Get average prediction for a class (average of all predictions for that class)
  Map<String, double> _getAveragePredictionsForAllClasses() {
    Map<String, List<double>> predictions = {};

    // Initialize all classes
    for (String className in allClasses) {
      predictions[className] = [];
    }

    // Collect all predictions
    for (DetectionItem item in detectionHistory) {
      if (item.allPredictions.isNotEmpty) {
        for (String className in allClasses) {
          if (item.allPredictions.containsKey(className)) {
            predictions[className]!.add(item.allPredictions[className]!);
          }
        }
      }
    }

    // Calculate averages
    Map<String, double> averages = {};
    for (String className in allClasses) {
      if (predictions[className]!.isNotEmpty) {
        double avg = predictions[className]!.reduce((a, b) => a + b) /
            predictions[className]!.length;
        averages[className] = avg;
      } else {
        averages[className] = 0.0;
      }
    }

    return averages;
  }

  // Get colors for the chart bars
  Color _getColorForClass(int index) {
    final colors = [
      const Color(0xFFFF6B6B),
      const Color(0xFF4ECDC4),
      const Color(0xFF45B7D1),
      const Color(0xFFFFA07A),
      const Color(0xFF98D8C8),
      const Color(0xFFF7DC6F),
      const Color(0xFFBB8FCE),
      const Color(0xFF85C1E2),
      const Color(0xFFF8B88B),
      const Color(0xFF82E0AA),
    ];
    return colors[index % colors.length];
  }

  @override
  Widget build(BuildContext context) {
    final avgConfidences = _getAverageConfidencePerClass();
    final avgPredictions = _getAveragePredictionsForAllClasses();
    final totalDetections =
        detectionHistory.where((item) => !item.isUnknown).length;

    // Find max confidence for chart scaling
    double maxConfidence = avgConfidences.values.isEmpty
        ? 100
        : (avgConfidences.values.reduce((a, b) => a > b ? a : b) + 10);
    maxConfidence = (maxConfidence > 100) ? 100 : maxConfidence;

    return Scaffold(
      body: Container(
        constraints: BoxConstraints(
          minHeight: MediaQuery.of(context).size.height, // 👈 IMPORTANT
        ),
        decoration: const BoxDecoration(
          image: DecorationImage(
            image: AssetImage(
              'assets/Midjourney_  Vibrant rural landscape with sustainable farming and renewable energy_.jfif',
            ),
            fit: BoxFit.cover,
          ),
        ),
        child: SafeArea(
          child: SingleChildScrollView(
            child: Padding(
              padding: const EdgeInsets.all(24.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const SizedBox(height: 20),
                  // Header
                  const Text(
                    'Analytics',
                    style: TextStyle(
                      fontSize: 24,
                      fontWeight: FontWeight.w700,
                      color: Color(0xFF1A1A1A),
                      letterSpacing: -0.5,
                    ),
                  ),
                  const SizedBox(height: 4),
                  const Text(
                    'Detection statistics by class',
                    style: TextStyle(
                      fontSize: 13,
                      color: Color(0xFF666666),
                      fontWeight: FontWeight.w400,
                    ),
                  ),
                  const SizedBox(height: 24),

                  // Total Detections Card
                  Container(
                    padding: const EdgeInsets.all(20),
                    decoration: BoxDecoration(
                      gradient: LinearGradient(
                        colors: [
                          Colors.white,
                          const Color(0xFFF8FFF8).withOpacity(0.5),
                        ],
                        begin: Alignment.topLeft,
                        end: Alignment.bottomRight,
                      ),
                      borderRadius: BorderRadius.circular(20),
                      border: Border.all(
                        color: const Color(0xFF2E5F4F).withOpacity(0.1),
                        width: 1,
                      ),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.black.withOpacity(0.08),
                          blurRadius: 16,
                          offset: const Offset(0, 4),
                        ),
                      ],
                    ),
                    child: Row(
                      children: [
                        Container(
                          padding: const EdgeInsets.all(12),
                          decoration: BoxDecoration(
                            gradient: LinearGradient(
                              colors: [
                                const Color(0xFF2E5F4F).withOpacity(0.9),
                                const Color(0xFF1A3A32),
                              ],
                              begin: Alignment.topLeft,
                              end: Alignment.bottomRight,
                            ),
                            borderRadius: BorderRadius.circular(16),
                            boxShadow: [
                              BoxShadow(
                                color: const Color(0xFF2E5F4F).withOpacity(0.2),
                                blurRadius: 8,
                                offset: const Offset(0, 2),
                              ),
                            ],
                          ),
                          child: const Icon(
                            Icons.trending_up_rounded,
                            color: Colors.white,
                            size: 28,
                          ),
                        ),
                        const SizedBox(width: 16),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                'Total Detections',
                                style: TextStyle(
                                  fontSize: 12,
                                  fontWeight: FontWeight.w600,
                                  color: Colors.grey.shade600,
                                  letterSpacing: 0.5,
                                ),
                              ),
                              const SizedBox(height: 4),
                              Text(
                                totalDetections.toString(),
                                style: const TextStyle(
                                  fontSize: 28,
                                  fontWeight: FontWeight.w700,
                                  color: Color(0xFF1A1A1A),
                                ),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),

                  const SizedBox(height: 24),

                  // Detections per Class - Bar Chart
                  Container(
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(20),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.black.withOpacity(0.08),
                          blurRadius: 16,
                          offset: const Offset(0, 4),
                        ),
                      ],
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Padding(
                          padding: EdgeInsets.all(20.0),
                          child: Text(
                            'Detections per Class',
                            style: TextStyle(
                              fontSize: 16,
                              fontWeight: FontWeight.w700,
                              color: Color(0xFF1A1A1A),
                            ),
                          ),
                        ),
                        detectionHistory.isEmpty
                            ? Center(
                                child: Padding(
                                  padding: const EdgeInsets.symmetric(
                                      vertical: 40.0),
                                  child: Text(
                                    'No detections yet',
                                    style: TextStyle(
                                      fontSize: 14,
                                      color: Colors.grey.shade600,
                                    ),
                                  ),
                                ),
                              )
                            : Padding(
                                padding: const EdgeInsets.all(20.0),
                                child: Stack(
                                  children: [
                                    SizedBox(
                                      height: 300,
                                      child: BarChart(
                                        BarChartData(
                                          maxY: 40,
                                          barTouchData: BarTouchData(
                                            enabled: true,
                                            touchTooltipData:
                                                BarTouchTooltipData(
                                              tooltipBgColor: Colors.black87,
                                              tooltipRoundedRadius: 8,
                                              getTooltipItem: (group,
                                                  groupIndex, rod, rodIndex) {
                                                return BarTooltipItem(
                                                  allClasses[groupIndex]
                                                      .toUpperCase(),
                                                  const TextStyle(
                                                    color: Colors.white,
                                                    fontWeight: FontWeight.bold,
                                                    fontSize: 12,
                                                  ),
                                                  children: [
                                                    TextSpan(
                                                      text:
                                                          '\n${rod.toY.toInt()} detections',
                                                      style: const TextStyle(
                                                        color: Colors.white70,
                                                        fontSize: 11,
                                                      ),
                                                    ),
                                                  ],
                                                );
                                              },
                                            ),
                                          ),
                                          gridData: FlGridData(
                                            show: true,
                                            drawVerticalLine: false,
                                            horizontalInterval: 5,
                                            getDrawingHorizontalLine: (value) {
                                              return FlLine(
                                                color: Colors.grey.shade100,
                                                strokeWidth: 1,
                                              );
                                            },
                                          ),
                                          titlesData: FlTitlesData(
                                            show: true,
                                            topTitles: const AxisTitles(
                                              sideTitles:
                                                  SideTitles(showTitles: false),
                                            ),
                                            rightTitles: const AxisTitles(
                                              sideTitles:
                                                  SideTitles(showTitles: false),
                                            ),
                                            bottomTitles: AxisTitles(
                                              sideTitles: SideTitles(
                                                showTitles: true,
                                                getTitlesWidget: (value, meta) {
                                                  int index = value.toInt();
                                                  if (index >= 0 &&
                                                      index <
                                                          allClasses.length) {
                                                    return Padding(
                                                      padding:
                                                          const EdgeInsets.only(
                                                              top: 8.0),
                                                      child: Text(
                                                        allClasses[index]
                                                            .substring(0, 3)
                                                            .toUpperCase(),
                                                        style: TextStyle(
                                                          color: Colors
                                                              .grey.shade600,
                                                          fontSize: 11,
                                                          fontWeight:
                                                              FontWeight.w600,
                                                        ),
                                                      ),
                                                    );
                                                  }
                                                  return const SizedBox();
                                                },
                                                reservedSize: 40,
                                              ),
                                            ),
                                            leftTitles: AxisTitles(
                                              sideTitles: SideTitles(
                                                showTitles: true,
                                                getTitlesWidget: (value, meta) {
                                                  return Text(
                                                    '${value.toInt()}',
                                                    style: TextStyle(
                                                      color:
                                                          Colors.grey.shade600,
                                                      fontSize: 10,
                                                    ),
                                                  );
                                                },
                                                reservedSize: 30,
                                              ),
                                            ),
                                          ),
                                          barGroups: List.generate(
                                            allClasses.length,
                                            (index) {
                                              int count =
                                                  _getDetectionCountPerClass()[
                                                          allClasses[index]] ??
                                                      0;
                                              return BarChartGroupData(
                                                x: index,
                                                barRods: [
                                                  BarChartRodData(
                                                    toY: count.toDouble(),
                                                    color: _getColorForClass(
                                                        index),
                                                    width: 18,
                                                    borderRadius:
                                                        const BorderRadius.only(
                                                      topLeft:
                                                          Radius.circular(6),
                                                      topRight:
                                                          Radius.circular(6),
                                                    ),
                                                  ),
                                                ],
                                              );
                                            },
                                          ),
                                        ),
                                      ),
                                    ),
                                    SizedBox(
                                      height: 300,
                                      child: Padding(
                                        padding: const EdgeInsets.only(
                                            left: 30.0, right: 20.0),
                                        child: Row(
                                          mainAxisAlignment:
                                              MainAxisAlignment.spaceEvenly,
                                          crossAxisAlignment:
                                              CrossAxisAlignment.end,
                                          children: List.generate(
                                            allClasses.length,
                                            (index) {
                                              int count =
                                                  _getDetectionCountPerClass()[
                                                          allClasses[index]] ??
                                                      0;
                                              double barHeight =
                                                  (count / 40) * 250;
                                              return SizedBox(
                                                width: 18,
                                                child: Column(
                                                  mainAxisAlignment:
                                                      MainAxisAlignment.end,
                                                  children: [
                                                    if (count > 0)
                                                      Container(
                                                        padding:
                                                            const EdgeInsets
                                                                .symmetric(
                                                          horizontal: 6,
                                                          vertical: 2,
                                                        ),
                                                        decoration:
                                                            BoxDecoration(
                                                          color:
                                                              _getColorForClass(
                                                                  index),
                                                          borderRadius:
                                                              BorderRadius
                                                                  .circular(4),
                                                        ),
                                                        child: Text(
                                                          count.toString(),
                                                          style:
                                                              const TextStyle(
                                                            color: Colors.white,
                                                            fontSize: 10,
                                                            fontWeight:
                                                                FontWeight.w700,
                                                          ),
                                                        ),
                                                      ),
                                                  ],
                                                ),
                                              );
                                            },
                                          ),
                                        ),
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                      ],
                    ),
                  ),

                  const SizedBox(height: 32),

                  // Detection History Section
                  Container(
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(20),
                      border: Border.all(
                        color: Colors.grey.shade300,
                        width: 1,
                      ),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.black.withOpacity(0.08),
                          blurRadius: 16,
                          offset: const Offset(0, 4),
                        ),
                      ],
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Padding(
                          padding: const EdgeInsets.all(20.0),
                          child: const Text(
                            'Detection History',
                            style: TextStyle(
                              fontSize: 16,
                              fontWeight: FontWeight.w700,
                              color: Color(0xFF1A1A1A),
                            ),
                          ),
                        ),
                        detectionHistory.isEmpty
                            ? Center(
                                child: Padding(
                                  padding: const EdgeInsets.symmetric(
                                      vertical: 40.0),
                                  child: Text(
                                    'No detection history available',
                                    style: TextStyle(
                                      fontSize: 14,
                                      color: Colors.grey.shade600,
                                    ),
                                  ),
                                ),
                              )
                            : ListView.builder(
                                shrinkWrap: true,
                                physics: const NeverScrollableScrollPhysics(),
                                itemCount: detectionHistory.length,
                                itemBuilder: (context, index) {
                                  final item = detectionHistory[index];
                                  return ListTile(
                                    leading: Image.file(
                                      File(item.imagePath),
                                      width: 50,
                                      height: 50,
                                      fit: BoxFit.cover,
                                    ),
                                    title: Text(item.label),
                                    subtitle: Text(
                                      'Confidence: ${item.confidence.toStringAsFixed(2)}%\nTime: ${item.timestamp}',
                                    ),
                                  );
                                },
                              ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 24),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  String _formatDateTime(DateTime dateTime) {
    final now = DateTime.now();
    final difference = now.difference(dateTime);
    if (difference.inMinutes < 1) {
      return 'Just now';
    } else if (difference.inHours < 1) {
      return '${difference.inMinutes}m ago';
    } else if (difference.inDays < 1) {
      return '${difference.inHours}h ago';
    } else if (difference.inDays == 1) {
      return 'Yesterday';
    } else {
      return '${dateTime.day}/${dateTime.month}/${dateTime.year}';
    }
  }
}

class AboutPage extends StatelessWidget {
  const AboutPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          image: DecorationImage(
            image: AssetImage(
              'assets/Midjourney_  Vibrant rural landscape with sustainable farming and renewable energy_.jfif',
            ),
            fit: BoxFit.cover,
          ),
        ),
        child: SafeArea(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(24.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const SizedBox(height: 20),
                const Text(
                  'About',
                  style: TextStyle(
                    fontSize: 24,
                    fontWeight: FontWeight.w700,
                    color: Color(0xFF1A1A1A),
                    letterSpacing: -0.5,
                  ),
                ),
                const SizedBox(height: 4),
                const Text(
                  'Application information',
                  style: TextStyle(
                    fontSize: 13,
                    color: Color(0xFF666666),
                    fontWeight: FontWeight.w400,
                  ),
                ),
                const SizedBox(height: 32),
                Container(
                  padding: const EdgeInsets.all(32),
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      colors: [
                        Colors.white,
                        const Color(0xFFF8FFF8).withOpacity(0.5),
                      ],
                      begin: Alignment.topLeft,
                      end: Alignment.bottomRight,
                    ),
                    borderRadius: BorderRadius.circular(24),
                    border: Border.all(
                      color: const Color(0xFF2E5F4F).withOpacity(0.1),
                      width: 1,
                    ),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withOpacity(0.08),
                        blurRadius: 16,
                        offset: const Offset(0, 4),
                      ),
                    ],
                  ),
                  child: Column(
                    children: [
                      Container(
                        padding: const EdgeInsets.all(20),
                        decoration: BoxDecoration(
                          gradient: LinearGradient(
                            colors: [
                              const Color(0xFF2E5F4F).withOpacity(0.9),
                              const Color(0xFF1A3A32),
                            ],
                            begin: Alignment.topLeft,
                            end: Alignment.bottomRight,
                          ),
                          borderRadius: BorderRadius.circular(20),
                          boxShadow: [
                            BoxShadow(
                              color: const Color(0xFF2E5F4F).withOpacity(0.2),
                              blurRadius: 12,
                              offset: const Offset(0, 4),
                            ),
                          ],
                        ),
                        child: const Icon(
                          Icons.spa_outlined,
                          size: 48,
                          color: Colors.white,
                        ),
                      ),
                      const SizedBox(height: 24),
                      const Text(
                        'RootScan AI',
                        style: TextStyle(
                          fontSize: 24,
                          fontWeight: FontWeight.w700,
                          color: Color(0xFF1A1A1A),
                          letterSpacing: -0.3,
                        ),
                      ),
                      const SizedBox(height: 8),
                      Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 16,
                          vertical: 6,
                        ),
                        decoration: BoxDecoration(
                          gradient: LinearGradient(
                            colors: [
                              const Color(0xFF2E5F4F).withOpacity(0.08),
                              const Color(0xFF1A3A32).withOpacity(0.04),
                            ],
                          ),
                          borderRadius: BorderRadius.circular(20),
                          border: Border.all(
                            color: const Color(0xFF2E5F4F).withOpacity(0.1),
                          ),
                        ),
                        child: const Text(
                          'Version 1.0.0',
                          style: TextStyle(
                            fontSize: 12,
                            color: Color(0xFF666666),
                            fontWeight: FontWeight.w500,
                          ),
                        ),
                      ),
                      const SizedBox(height: 24),
                      const Text(
                        'An intelligent mobile application powered by TensorFlow Lite for accurate identification and classification of root crops commonly found in markets.',
                        textAlign: TextAlign.center,
                        style: TextStyle(
                          fontSize: 14,
                          color: Color(0xFF666666),
                          height: 1.6,
                          letterSpacing: 0.1,
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 24),
                Container(
                  padding: const EdgeInsets.all(24),
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      colors: [
                        Colors.white,
                        const Color(0xFFF8FFF8).withOpacity(0.5),
                      ],
                      begin: Alignment.topLeft,
                      end: Alignment.bottomRight,
                    ),
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(
                      color: const Color(0xFF2E5F4F).withOpacity(0.1),
                      width: 1,
                    ),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withOpacity(0.08),
                        blurRadius: 16,
                        offset: const Offset(0, 4),
                      ),
                    ],
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        'Key Features',
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.w700,
                          color: Color(0xFF1A1A1A),
                          letterSpacing: -0.2,
                        ),
                      ),
                      const SizedBox(height: 20),
                      _buildFeatureItem(
                        Icons.precision_manufacturing_outlined,
                        'High Accuracy Detection',
                        'Advanced AI model for precise identification',
                      ),
                      _buildFeatureItem(
                        Icons.bolt_outlined,
                        'Real-time Processing',
                        'Instant analysis and results',
                      ),
                      _buildFeatureItem(
                        Icons.cloud_off_outlined,
                        'Offline Capability',
                        'Works without internet connection',
                      ),
                      _buildFeatureItem(
                        Icons.history_rounded,
                        'Detection History',
                        'Track and review past scans',
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 24),
                Container(
                  padding: const EdgeInsets.all(20),
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      colors: [
                        const Color(0xFFFFFBF5).withOpacity(0.8),
                        const Color(0xFFFFF8F0),
                      ],
                      begin: Alignment.topLeft,
                      end: Alignment.bottomRight,
                    ),
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(
                      color: const Color(0xFFFFE0B2),
                    ),
                  ),
                  child: Row(
                    children: [
                      Container(
                        padding: const EdgeInsets.all(10),
                        decoration: BoxDecoration(
                          color: const Color(0xFFFFECB3),
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: const Icon(
                          Icons.tips_and_updates_outlined,
                          color: Color(0xFFE65100),
                          size: 20,
                        ),
                      ),
                      const SizedBox(width: 16),
                      const Expanded(
                        child: Text(
                          'Designed for agricultural professionals and market vendors',
                          style: TextStyle(
                            fontSize: 13,
                            color: Color(0xFF5D4037),
                            fontWeight: FontWeight.w500,
                            height: 1.4,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildFeatureItem(IconData icon, String title, String subtitle) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 16.0),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(
              color: const Color(0xFFF5F5F5),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Icon(
              icon,
              size: 20,
              color: const Color(0xFF1A1A1A),
            ),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: const TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.w600,
                    color: Color(0xFF1A1A1A),
                    letterSpacing: -0.1,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  subtitle,
                  style: TextStyle(
                    fontSize: 12,
                    color: Colors.grey.shade600,
                    height: 1.3,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
