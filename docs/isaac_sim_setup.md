# Isaac Sim Integration Guide

This guide explains how to test the SMAIT HRI system using NVIDIA Isaac Sim simulation.

## Prerequisites

- Ubuntu 22.04 (required for Isaac Sim + ROS 2 Humble compatibility)
- NVIDIA GPU with RTX support (RTX 2070 or higher recommended)
- NVIDIA Driver 525.60+ 
- 32GB+ RAM recommended
- Isaac Sim 4.x (free, Apache 2.0 license)

## Installation

### 1. Install Isaac Sim

```bash
# Download Isaac Sim from NVIDIA
# https://developer.nvidia.com/isaac/sim

# Or use the Omniverse Launcher
# https://www.nvidia.com/en-us/omniverse/download/

# Verify installation
cd ~/.local/share/ov/pkg/isaac-sim-4.2.0
./isaac-sim.sh --help
```

### 2. Install ROS 2 Humble

```bash
# Add ROS 2 repository
sudo apt update && sudo apt install -y software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
    | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS 2 Humble
sudo apt update
sudo apt install -y ros-humble-desktop ros-humble-cv-bridge ros-humble-image-transport

# Source ROS 2
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### 3. Configure Isaac Sim ROS 2 Bridge

```bash
# Source ROS 2 BEFORE launching Isaac Sim
source /opt/ros/humble/setup.bash

# For multi-machine setups, configure FastDDS
export FASTRTPS_DEFAULT_PROFILES_FILE=~/.ros/fastdds.xml

# Launch Isaac Sim with ROS 2 support
cd ~/.local/share/ov/pkg/isaac-sim-4.2.0
./isaac-sim.sh
```

## Setting Up the Test Scene

### 1. Create a Simple HRI Test Scene

In Isaac Sim:

1. **Create New Stage**: File → New
2. **Add Floor**: Create → Physics → Ground Plane
3. **Add Robot**: 
   - Isaac Assets → Robots → search "Carter" or your robot
   - Drag into scene
4. **Add Camera**:
   - Create → Camera
   - Position to view interaction area
   - Note the prim path (e.g., `/World/Camera`)
5. **Add Human Character**:
   - Isaac Assets → Characters → search "human" or "person"
   - Position facing the robot
6. **Save Scene**: File → Save As → `smait_test_scene.usd`

### 2. Configure Camera for ROS 2 Publishing

Using the Action Graph:

1. Window → Visual Scripting → Action Graph
2. Create new Action Graph
3. Add nodes:
   ```
   On Playback Tick
        │
        ▼
   Isaac Create Render Product (camera_prim=/World/Camera)
        │
        ▼
   ROS2 Camera Helper (type=rgb)
        │
        ▼
   ROS2 Publish Image
   ```
4. Set topic name to `/smait/camera/image_raw`

### 3. Add Human Animation (Optional)

For testing lip-sync detection:

1. Select the human character
2. In Properties → Animation:
   - Add "talking" animation
   - Enable loop
3. This provides animated faces for testing the vision pipeline

## Running SMAIT with Isaac Sim

### Method 1: Using ROS 2 Bridge (Recommended)

Terminal 1 - Isaac Sim:
```bash
source /opt/ros/humble/setup.bash
cd ~/.local/share/ov/pkg/isaac-sim-4.2.0
./isaac-sim.sh
# Open your test scene and press Play
```

Terminal 2 - SMAIT:
```bash
source /opt/ros/humble/setup.bash
cd ~/smait_hri_v2

# Configure for Isaac Sim mode
export SMAIT_MODE=ros2
export SMAIT_CAMERA_TOPIC=/smait/camera/image_raw

# Run SMAIT
python -m smait.main
```

### Method 2: Direct Integration (Advanced)

For tighter integration, SMAIT can run inside Isaac Sim's Python environment:

```python
# smait_isaac_extension.py
import omni.ext
from smait.core.config import Config, DeploymentMode, set_config
from smait.main import HRISystem

class SmaitHRIExtension(omni.ext.IExt):
    def on_startup(self, ext_id):
        print("[SMAIT] Extension starting...")
        
        # Configure for Isaac Sim
        config = Config()
        config.mode = DeploymentMode.ISAAC_SIM
        set_config(config)
        
        # Initialize system
        self.hri = HRISystem(config)
        
    def on_shutdown(self):
        print("[SMAIT] Extension shutting down...")
```

## Testing Scenarios

### Scenario 1: Basic Conversation

1. Start Isaac Sim with test scene
2. Press Play in Isaac Sim
3. Start SMAIT
4. Speak to test ASR and response generation
5. Verify camera feed is being processed

### Scenario 2: Multi-Person (Phase 2)

1. Add multiple human characters to scene
2. Animate one as "talking"
3. Test that SMAIT correctly identifies the active speaker

### Scenario 3: Noise Robustness (Phase 3)

1. Add ambient audio sources to scene
2. Test speech recognition accuracy
3. Verify RAVEN audio-visual enhancement (if enabled)

## Troubleshooting

### Camera Not Publishing

```bash
# Check ROS 2 topics
ros2 topic list
ros2 topic echo /smait/camera/image_raw --once
```

### Poor Performance

- Reduce camera resolution in Isaac Sim
- Disable ray tracing (use "RTX - Real-Time" mode)
- Close other GPU-intensive applications

### ROS 2 Bridge Not Connecting

```bash
# Verify ROS 2 is sourced
echo $ROS_DISTRO  # Should show "humble"

# Check network configuration
export ROS_LOCALHOST_ONLY=1

# Restart Isaac Sim with ROS 2 environment
source /opt/ros/humble/setup.bash
./isaac-sim.sh
```

## Sample Scene File

A pre-configured test scene is available at:
```
isaac_sim/scenes/smait_test_scene.usd
```

Load it in Isaac Sim:
1. File → Open
2. Navigate to the scene file
3. Press Play

## Performance Tips

1. **Use RTX-Direct mode** for faster rendering
2. **Reduce physics substeps** if not testing navigation
3. **Limit camera FPS** to 15-30 for HRI (don't need 60fps)
4. **Use headless mode** for automated testing:
   ```bash
   ./isaac-sim.sh --headless
   ```

## Next Steps

- [ROS 2 Node Integration](ros2_setup.md)
- [Light-ASD Setup](light_asd_setup.md)
- [Behavior Tree Configuration](behavior_tree_setup.md)
