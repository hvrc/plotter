# Agents Documentation

## Project Overview
This is a modular plotter system that generates SVG plots from images and controls an AxiDraw plotter. The system is designed to be scalable with pluggable algorithms and interfaces, featuring a sophisticated settings management system and multiple artistic rendering algorithms.

---

## Core Agents

### 1. Interface Agent
**File**: `scripts/interface.py` (697 lines)
**Purpose**: Comprehensive command-line interface for user interaction

**Key Features**:
- Dynamic menu system that adapts to algorithm capabilities
- Procedural vs image-based algorithm detection
- Multi-color plot support with individual SVG files
- SVG merging capabilities
- Real-time progress feedback
- Input validation and error handling

**Core Methods**:
- `main_menu()`: Primary navigation hub with 8 main options
- `generate_plot_menu()`: Handles both image-based and procedural generation
- `run_plot_menu()`: Multi-SVG plotting with preview mode
- `plotter_control_menu()`: Manual AxiDraw control interface
- `algorithm_settings_menu()`: Runtime algorithm parameter adjustment
- `plotter_settings_menu()`: Hardware configuration management
- `merge_svgs_menu()`: SVG file combination tool

**Advanced Capabilities**:
- Automatic procedural detection via `_algorithm_is_procedural()`
- Type-aware input parsing via `_parse_typed_value()`
- Selection UI with `_select_from_list()` helper
- Algorithm switching with persistent state management

---

### 2. Manager Agent
**File**: `scripts/manager.py` (381 lines)
**Purpose**: Sophisticated file and database management

**Key Features**:
- Path-based file operations with type safety
- Multi-SVG file handling per plot
- Metadata preservation with JSON storage
- SVG merging with namespace handling
- Directory structure validation

**Core Methods**:
- `list_images()`: Filtered image discovery with extension support
- `save_plot()`: Multi-format plot storage (SVG, preview, original, metadata)
- `get_plot_svgs()`: Multi-SVG file enumeration with sorting
- `merge_svgs()`: XML namespace-aware SVG combination
- `get_database_stats()`: Usage statistics and reporting

**Data Management**:
- Supports single and multi-color plot storage
- Automatic directory creation with validation
- File copying with metadata preservation
- Empty plot cleanup functionality

---

### 3. Plot Generation Agents

#### Algorithm Base Class
**File**: `algorithms/base.py` (62 lines)
**Purpose**: Abstract interface for all plot algorithms

**Interface Contract**:
- `get_algorithm_name()`: Algorithm identifier
- `get_algorithm_description()`: Human-readable description
- `is_procedural()`: Image requirement detection
- `generate_paths()`: Path generation with metadata
- `generate_svg()`: SVG creation from paths
- `generate_preview()`: PNG preview generation

#### Algorithm Registry
**File**: `algorithms/__init__.py` (146 lines)
**Purpose**: Dynamic algorithm discovery and management

**Features**:
- Automatic algorithm discovery via reflection
- Registry pattern with normalization
- Class instantiation with configuration
- Error handling for malformed algorithms

#### Waves Algorithm
**File**: `algorithms/waves.py` (484 lines)
**Purpose**: Wave-based plot generation with sine modulation

**Advanced Features**:
- Multi-color support with k-means clustering
- Horizontal and vertical drawing directions
- Serpentine path optimization
- Color quantization with mask support
- Configurable wave parameters (amplitude, frequency, contrast)

**Configuration**:
```python
DEFAULT_CONFIG = {
    'output_width_inches': 4.5,
    'output_height_inches': 6.0,
    'margin_inches': 0.125,
    'n_rows': 100,
    'contrast_power': 1.5,
    'amplitude_scale': 1,
    'frequency_scale': 1.75,
    'white_threshold': 250,
    'use_serpentine': False,
    'draw_direction': 'horizontal',
    'numColors': 1,
    'color_mask': []
}
```

**Key Methods**:
- `_quantize_colors()`: Color reduction using PIL quantization
- `_create_color_mask()`: Binary mask generation for color separation
- `_generate_horizontal_paths()`: Left-to-right wave generation
- `_generate_vertical_paths()`: Top-to-bottom wave generation

#### Additional Algorithms

**Circles** (`algorithms/circles.py`):
- Concentric circle generation with brightness modulation
- Configurable circle count and point density
- Squiggle generation based on image analysis

**Spiral** (`algorithms/spiral.py`):
- Procedural attractor-based spiral generation
- No input image required (fully procedural)
- Configurable attractor points and particle physics

**Fabric** (`algorithms/fabric.py`):
- Grid-based displacement with noise functions
- Support for image-driven and random displacement
- Weave pattern generation with directional control

**Fish** (`algorithms/fish.py`):
- Circular pattern generation with radius-based features
- High-resolution point sampling

**Sphere** (`algorithms/sphere.py`):
- 3D sphere projection with latitude/longitude lines
- Flow line generation on spherical surfaces
- Hidden line removal options

**Features** (`algorithms/features.py`):
- Feature-based pattern generation
- Configurable feature types and distributions

**Terrain** (`algorithms/terrain.py`):
- 3D terrain generation with Perlin noise
- Isometric projection with rotation controls

---

### 4. Plotter Agent
**File**: `scripts/plotter.py` (285 lines)
**Purpose**: AxiDraw hardware control with configuration management

**Key Features**:
- Graceful degradation when pyaxidraw unavailable
- Configuration validation and defaults
- Preview mode operation
- Manual command execution
- Connection testing capabilities

**Core Classes**:
- `PlotterController`: Hardware interface with configuration
- `PlotterSettings`: Persistent settings management

**Configuration**:
```python
DEFAULT_CONFIG = {
    'pen_pos_up': 100,
    'pen_pos_down': 30,
    'auto_rotate': False,
    'speed_pendown': 25,
    'speed_penup': 75,
    'accel': 75
}
```

**Methods**:
- `plot_file()`: SVG plotting with preview support
- `execute_command()`: Manual control (disable_xy, walk_home, raise_pen, lower_pen)
- `test_connection()`: Hardware connectivity validation
- `_configure_axidraw()`: Instance configuration application

---

### 5. Settings Agent
**File**: `scripts/settings.py` (381 lines)
**Purpose**: Centralized configuration management with persistence

**Advanced Features**:
- Hierarchical settings with defaults and overrides
- Algorithm validation and typo correction
- Automatic settings migration
- Import/export functionality
- Deep copying for immutability

**Configuration Hierarchy**:
1. Default settings (`settings/default.json`)
2. User settings (`settings/settings.json`)
3. Runtime modifications

**Key Methods**:
- `get_algorithm_settings()`: Algorithm-specific configuration
- `set_algorithm_setting()`: Individual parameter updates
- `update_algorithm_settings()`: Batch configuration changes
- `_validate_current_algorithm()`: Algorithm name correction
- `export_settings()`/`import_settings()`: Configuration portability

**Validation Features**:
- Common typo correction (e.g., 'cirlces' → 'circles')
- Fallback to available algorithms
- Auto-save on validation fixes

---

## Agent Communication Flow

```
User Input → Interface Agent → Settings Agent → Algorithm Selection
                ↓                    ↓              ↓
            Manager Agent ← File Operations ← Path Generation
                ↓                    ↓              ↓
            SVG Creation → Preview Generation → Plot Storage
                ↓                    ↓              ↓
            Plotter Agent ← Hardware Control → Physical Output
```

---

## Configuration System

### Settings Structure
```json
{
  "plotter": {
    "pen_pos_up": 100,
    "pen_pos_down": 30,
    "auto_rotate": false,
    "speed_pendown": 25,
    "speed_penup": 75,
    "accel": 75
  },
  "algorithm": {
    "current_algorithm": "waves",
    "waves": { /* algorithm-specific config */ },
    "circles": { /* algorithm-specific config */ },
    // ... other algorithms
  }
}
```

### Algorithm Configuration
Each algorithm maintains its own configuration section with:
- Canvas dimensions (width, height, margins)
- DPI and resolution settings
- Algorithm-specific parameters
- Export and rendering options

---

## Main Entry Point
**File**: `main.py` (64 lines)
**Purpose**: Application initialization with dependency injection

**Initialization Sequence**:
1. Path setup for module imports
2. Settings manager initialization
3. Database manager creation
4. Plotter controller configuration
5. Interface agent instantiation
6. Error handling and graceful shutdown

**Features**:
- AxiDraw availability checking
- Graceful degradation for missing dependencies
- Exception handling with user-friendly messages
- Clean shutdown on keyboard interrupt

---

## Database Structure

### Directory Layout
```
database/
├── images/          # Source images (PNG, JPG, JPEG, etc.)
└── plots/           # Generated plots
    └── [plot_name]/
        ├── [plot_name].svg              # Primary SVG
        ├── [plot_name]_color1_XXXXXX.svg # Color-specific SVGs
        ├── [plot_name]_preview.png      # Preview image
        ├── [original_image]             # Source image copy
        └── metadata.json                # Generation metadata
```

### Metadata Schema
```json
{
  "algorithm": "waves",
  "algorithm_config": { /* algorithm settings */ },
  "total_width_px": 1350,
  "total_height_px": 1800,
  "output_width_inches": 4.5,
  "output_height_inches": 6.0,
  "numColors": 1,
  "dominant_colors": null
}
```

---

## Scalability Design

### Adding New Algorithms
1. Create class inheriting from `PlotAlgorithm`
2. Implement required abstract methods
3. Define `DEFAULT_CONFIG` with parameters
4. Place in `algorithms/` directory
5. Auto-discovery handles registration
6. Settings automatically include algorithm section

### Algorithm Interface Requirements
```python
class CustomAlgorithm(PlotAlgorithm):
    DEFAULT_CONFIG = { /* parameters */ }
    
    def get_algorithm_name(self) -> str:
        return "custom"
    
    def get_algorithm_description(self) -> str:
        return "Custom algorithm description"
    
    def is_procedural(self) -> bool:
        return False  # or True if no image needed
    
    def generate_paths(self, image_path: Optional[str] = None) -> Tuple[Any, Dict[str, Any]]:
        # Generate path data and metadata
        pass
    
    def generate_svg(self, paths: Any, metadata: Dict[str, Any]) -> Any:
        # Create SVG from paths
        pass
    
    def generate_preview(self, paths: Any, metadata: Dict[str, Any]) -> Image.Image:
        # Create preview PNG
        pass
```

---

## Advanced Features

### Multi-Color Support
- Automatic color quantization using k-means clustering
- Individual SVG files per color layer
- Color mask support for selective layer generation
- Hex color naming in SVG files

### SVG Merging
- XML namespace-aware combination
- Multiple file selection with comma separation
- Timestamped output naming
- Error handling for malformed SVGs

### Procedural Generation
- Algorithm detection via `is_procedural()`
- No input image requirement
- Consistent naming for overwriting
- Mixed mode support (e.g., fabric algorithm)

### Configuration Management
- Runtime parameter adjustment
- Persistent settings storage
- Validation and correction
- Import/export capabilities
- Default value management

---

## Agent Dependencies

### Required Libraries
- `pyaxidraw`: AxiDraw plotter control (optional)
- `PIL` (Pillow): Image processing and manipulation
- `xml.etree.ElementTree`: SVG parsing and generation
- `pathlib`: Modern path handling
- `typing`: Type hints and interfaces

### External Dependencies
- AxiDraw hardware (for plotter functionality)
- Image files (for image-based algorithms)
- Configuration files (JSON format)
- File system access for database operations

### Optional Dependencies
- `numpy`: Some algorithms may use for performance (not required)
- Additional PIL features for advanced image processing

---

## Future Agent Possibilities

### Web Interface Agent
- Flask/FastAPI-based REST API
- WebSocket support for real-time progress
- File upload with validation
- Remote plotter status monitoring
- Queue-based job management

### GUI Interface Agent
- Tkinter/PyQt desktop application
- Drag-and-drop file handling
- Real-time parameter adjustment with sliders
- Live preview updates
- Visual algorithm selection

### Batch Processing Agent
- Queue management system
- Background worker threads
- Progress tracking and resumption
- Error recovery and retry logic
- Automated parameter sweeps

### Cloud Storage Agent
- AWS S3/Google Drive integration
- Synchronization capabilities
- Backup and versioning
- Collaborative sharing
- Remote configuration management

### Machine Learning Agent
- Image preprocessing with computer vision
- Algorithm parameter optimization
- Style transfer capabilities
- Automatic algorithm selection
- Quality assessment and scoring