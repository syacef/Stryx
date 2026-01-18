#!/bin/bash
set -e # Exit immediately if any command fails

echo "=== Step 1: Pre-flight Checks ==="
# 1. Check for Nvidia Drivers
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. Please install Nvidia drivers first."
    exit 1
fi
echo "✅ Nvidia Drivers found."

# 2. Check for Nvidia Container Toolkit
if ! command -v nvidia-container-runtime &> /dev/null; then
    echo "Error: nvidia-container-runtime not found. Run: sudo apt-get install -y nvidia-container-toolkit"
    exit 1
fi
echo "✅ Nvidia Container Toolkit found."


echo "=== Step 2: Fixing K3s Configuration (Network + GPU) ==="
# 1. Delete any existing broken template to restore default networking
echo "Removing old/broken container templates..."
sudo rm -f /var/lib/rancher/k3s/agent/etc/containerd/config.toml.tmpl

# 2. Restart K3s to regenerate a clean, working default config
echo "Restarting K3s to regenerate default network config..."
sudo systemctl restart k3s
# Wait for the file to be generated
timeout 30s bash -c 'until [ -f /var/lib/rancher/k3s/agent/etc/containerd/config.toml ]; do sleep 1; done'

# 3. Create the new template based on the WORKING default config
# (This preserves the CNI/Network settings that were missing before)
echo "Creating new safe GPU template..."
sudo cp /var/lib/rancher/k3s/agent/etc/containerd/config.toml /var/lib/rancher/k3s/agent/etc/containerd/config.toml.tmpl

# 4. Append the Nvidia Runtime configuration to the end
sudo bash -c 'cat <<EOF >> /var/lib/rancher/k3s/agent/etc/containerd/config.toml.tmpl

# --- NVIDIA CONFIGURATION INJECTED BY SETUP SCRIPT ---
[plugins.cri.containerd.runtimes."nvidia"]
  runtime_type = "io.containerd.runc.v2"
[plugins.cri.containerd.runtimes."nvidia".options]
  BinaryName = "/usr/bin/nvidia-container-runtime"
  SystemdCgroup = true
EOF'

# 5. Restart K3s to apply the GPU config
echo "Restarting K3s to apply GPU configuration..."
sudo systemctl restart k3s
sleep 10 # Give it a moment to stabilize


echo "=== Step 3: Installing Kubernetes Resources ==="
# 1. Define the RuntimeClass
cat <<EOF | kubectl apply -f -
apiVersion: node.k8s.io/v1
kind: RuntimeClass
metadata:
  name: nvidia
handler: nvidia
EOF
echo "✅ RuntimeClass created."

# 2. Deploy the Nvidia Device Plugin (using the correct v0.16.2 URL)
# We delete it first to ensure a clean install if it exists in a bad state
kubectl delete ds nvidia-device-plugin-daemonset -n kube-system --ignore-not-found=true
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.16.2/deployments/static/nvidia-device-plugin.yml
echo "✅ Device Plugin deployed."

# 3. Patch the DaemonSet (Crucial for single-node or tainted clusters)
# This forces the plugin to use the 'nvidia' runtime and ignore taints
kubectl patch daemonset nvidia-device-plugin-daemonset -n kube-system -p '{"spec":{"template":{"spec":{"runtimeClassName":"nvidia"}}}}'
kubectl patch ds nvidia-device-plugin-daemonset -n kube-system --type='json' -p='[{"op": "add", "path": "/spec/template/spec/tolerations/-", "value": {"operator": "Exists"}}]'
echo "✅ DaemonSet patched."


echo "=== Step 4: Verification ==="
echo "Waiting for the Device Plugin to become Ready..."
kubectl rollout status daemonset nvidia-device-plugin-daemonset -n kube-system --timeout=60s

echo "Checking if Node advertises GPU..."
if kubectl describe node k8s | grep -q "nvidia.com/gpu"; then
    echo "🎉 SUCCESS: GPU detected on node!"
else
    echo "⚠️ WARNING: GPU not yet seen on node. It might take a few more seconds."
fi

echo "=== Setup Complete! ==="
echo "You can now run your test pod."