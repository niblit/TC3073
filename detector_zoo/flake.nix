{
  description = "A reproducible evasion and robustness engine for phishing/BEC detectors";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pkgs.python312
            pkgs.uv
            # Required C libraries for pre-compiled PyPI wheels (torch, tokenizers, etc.)
            pkgs.stdenv.cc.cc.lib
            pkgs.zlib
          ];

          shellHook = ''
            # Tell the dynamic linker where to find the C/C++ libraries
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [
              pkgs.stdenv.cc.cc.lib
              pkgs.zlib
            ]}:$LD_LIBRARY_PATH"

            # Expose the host NVIDIA driver (libcuda.so) on NixOS
            if [ -d /run/opengl-driver/lib ]; then
              export LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH"
            fi

            # Create a virtual environment if it doesn't exist
            if [ ! -d ".venv" ]; then
              echo "Setting up fast virtual environment via uv..."
              uv venv
              source .venv/bin/activate

              # Install all deps from requirements.txt
              uv pip install -r requirements.txt
            else
              source .venv/bin/activate
            fi

            # Expose CUDA/cuDNN libraries bundled by pip-installed nvidia-* packages
            # (e.g. nvidia-cudnn-cu13, nvidia-cu13, etc.)
            NVIDIA_LIBS=$(python -c '
import glob, site
dirs = glob.glob(site.getsitepackages()[0] + "/nvidia/*/lib")
print(":".join(dirs))
' 2>/dev/null)
            if [ -n "$NVIDIA_LIBS" ]; then
              export LD_LIBRARY_PATH="$NVIDIA_LIBS:$LD_LIBRARY_PATH"
            fi

            echo ""
            echo "Detector Zoo environment ready."
            python --version
          '';
        };
      }
    );
}
