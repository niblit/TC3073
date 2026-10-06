{
  description = "Degradation curve analysis: measuring detector robustness under adversarial perturbation";

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
            # Required C libraries for pre-compiled PyPI wheels
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

            export NLTK_DATA=$PWD/.nltk_data
            mkdir -p $NLTK_DATA

            # Create a virtual environment if it doesn't exist
            if [ ! -d ".venv" ]; then
              echo "Setting up fast virtual environment via uv..."
              uv venv
              source .venv/bin/activate

              # Install all deps: detector (sklearn, joblib, pandas) + attacker (nltk, spacy, sentence-transformers) + analysis (matplotlib)
              uv pip install \
                scikit-learn joblib pandas numpy \
                nltk spacy sentence-transformers \
                tensorflow tensorflow-hub gensim setuptools \
                matplotlib

              # Let spacy fetch the correct model version dynamically
              python -m spacy download en_core_web_sm
            else
              source .venv/bin/activate
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

            # NLTK setup
            python -c "import nltk; nltk.download('wordnet', download_dir='$NLTK_DATA', quiet=True); nltk.download('averaged_perceptron_tagger_eng', download_dir='$NLTK_DATA', quiet=True); nltk.download('stopwords', download_dir='$NLTK_DATA', quiet=True); nltk.download('punkt_tab', download_dir='$NLTK_DATA', quiet=True)"

            echo ""
            echo "Analysis environment ready."
            python --version
          '';
        };
      }
    );
}
