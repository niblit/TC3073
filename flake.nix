{
  description = "Unified reproducible evasion and robustness engine";

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
            pkgs.git
            pkgs.stdenv.cc.cc.lib
            pkgs.zlib
          ];

          shellHook = ''
            # Tell the dynamic linker where to find C/C++ libraries
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [
              pkgs.stdenv.cc.cc.lib
              pkgs.zlib
            ]}:$LD_LIBRARY_PATH"

            # Expose the host NVIDIA driver (libcuda.so) on NixOS
            if [ -d /run/opengl-driver/lib ]; then
              export LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH"
            fi

            # Explicitly tie repository paths to the Git root
            REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || echo "$PWD")"
            export NLTK_DATA="$REPO_ROOT/.nltk_data"
            mkdir -p "$NLTK_DATA"

            # Create virtual environment and install dependencies at repo root
            if [ ! -d "$REPO_ROOT/.venv" ]; then
              echo "Setting up unified virtual environment via uv..."
              uv venv "$REPO_ROOT/.venv"
              source "$REPO_ROOT/.venv/bin/activate"
              
              uv pip install -r "$REPO_ROOT/requirements.txt"
              
              # Fetch SpaCy model dynamically
              python -m spacy download en_core_web_sm
            else
              source "$REPO_ROOT/.venv/bin/activate"
            fi

            # Expose CUDA/cuDNN libraries bundled by pip-installed nvidia-* packages
            NVIDIA_LIBS=$(python -c '
import glob, site
dirs = glob.glob(site.getsitepackages()[0] + "/nvidia/*/lib")
print(":".join(dirs))
' 2>/dev/null)
            if [ -n "$NVIDIA_LIBS" ]; then
              export LD_LIBRARY_PATH="$NVIDIA_LIBS:$LD_LIBRARY_PATH"
            fi

            # NLTK setup (includes wordnet, POS taggers, and tokenizers)
            python -c "import nltk; nltk.download('wordnet', download_dir='$NLTK_DATA', quiet=True); nltk.download('averaged_perceptron_tagger_eng', download_dir='$NLTK_DATA', quiet=True); nltk.download('stopwords', download_dir='$NLTK_DATA', quiet=True); nltk.download('punkt_tab', download_dir='$NLTK_DATA', quiet=True)"

            echo ""
            echo "Unified Detector & Evasion environment ready."
            python --version
          '';
        };
      }
    );
}