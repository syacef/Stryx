# 1. Verify NVIDIA Drivers are loaded
nvidia-smi

# 2. Verify Container Toolkit is installed
# If missing: sudo apt-get install -y nvidia-container-toolkit
which nvidia-container-runtime


# 1. Create the template file (uses the safe, dynamic config)
sudo bash -c 'cat <<EOF > /var/lib/rancher/k3s/agent/etc/containerd/config.toml.tmpl
[plugins.opt]
  path = "{{ .NodeConfig.Containerd.Opt }}"
  [plugins.opt.containerd]
    snapshotter = "overlayfs"
    disable_snapshot_annotations = true
    snapshotter = "overlayfs"

[plugins.cri]
  stream_server_address = "127.0.0.1"
  stream_server_port = "10010"
  enable_selinux = {{ .NodeConfig.SELinux }}
  enable_unprivileged_ports = {{ .NodeConfig.AgentConfig.AllowUnprivilegedPorts }}
  enable_unprivileged_icmp = {{ .NodeConfig.AgentConfig.AllowUnprivilegedICMP }}

[plugins.cri.containerd]
  snapshotter = "overlayfs"
  disable_snapshot_annotations = true

[plugins.cri.containerd.runtimes.runc]
  runtime_type = "io.containerd.runc.v2"

# --- NVIDIA CONFIGURATION ---
[plugins.cri.containerd.runtimes."nvidia"]
  runtime_type = "io.containerd.runc.v2"
[plugins.cri.containerd.runtimes."nvidia".options]
  BinaryName = "/usr/bin/nvidia-container-runtime"
  SystemdCgroup = true
EOF'

# 2. Restart K3s to apply changes
sudo systemctl restart k3s


# 1. Create the RuntimeClass
cat <<EOF | kubectl apply -f -
apiVersion: node.k8s.io/v1
kind: RuntimeClass
metadata:
  name: nvidia
handler: nvidia
EOF

# 2. Deploy the NVIDIA Device Plugin
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml

# 3. Patch the Plugin to use the "nvidia" RuntimeClass
# (This is the critical fix for "Incompatible platform" errors)
kubectl patch daemonset nvidia-device-plugin-daemonset -n kube-system -p '{"spec":{"template":{"spec":{"runtimeClassName":"nvidia"}}}}'


# 1. Deploy a test pod requesting GPU
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: gpu-test
spec:
  restartPolicy: OnFailure
  runtimeClassName: nvidia
  containers:
    - name: cuda-vector-add
      image: "k8s.gcr.io/cuda-vector-add:v0.1"
      resources:
        limits:
          nvidia.com/gpu: 1
EOF

# 2. Wait for it to finish and check logs
sleep 10
kubectl logs gpu-test